import tensorflow as tf



class ProbRPROPOptimizer(tf.train.GradientDescentOptimizer):

    def __init__(self, learning_rate, name="ProbRPROP", mu=0.95, delta_0=0.1,
                 delta_min=10 ^(-6), delta_max=50,
                 eta_minus=0.5, eta_plus=1.2,eps=1e-8):
        super(ProbRPROPOptimizer, self).__init__(learning_rate, name=name)
        self._mu = mu
        self._lr = learning_rate
        self._delta_0 = delta_0
        self._delta_min = delta_min
        self._delta_max = delta_max
        self._eta_minus = eta_minus
        self._eta_plus = eta_plus
        self._eps=eps


    def minimize(self, loss, var_list=None):

        # Algo params as constant tensors
        mu = tf.convert_to_tensor(self._mu, dtype=tf.float32)
        delta_0=tf.convert_to_tensor(self._delta_0, dtype=tf.float32)
        delta_min=tf.convert_to_tensor(self._delta_min, dtype=tf.float32)
        delta_max=tf.convert_to_tensor(self._delta_max, dtype=tf.float32)
        eta_minus=tf.convert_to_tensor(self._eta_minus, dtype=tf.float32)
        eta_plus=tf.convert_to_tensor(self._eta_plus, dtype=tf.float32)

        if var_list is None:
            var_list = tf.trainable_variables()
            print(var_list)


        # Create and retrieve slot variables for delta , old_grad values
        # and old_dir (values of gradient changes)

        old_deltas = [self._get_or_make_slot(var,
                  tf.constant(0.1, tf.float32, var.get_shape()), "delta", "delta")
                  for var in var_list]
        old_grads = [self._get_or_make_slot(var,
                  tf.constant(0, tf.float32, var.get_shape()), "grads", "grads")
                  for var in var_list]
        # moving average estimation
        ms = [self._get_or_make_slot(var,
            tf.constant(0.0, tf.float32, var.get_shape()), "m", "m")
            for var in var_list]
        vs = [self._get_or_make_slot(var,
            tf.constant(0.0, tf.float32, var.get_shape()), "v", "v")
            for var in var_list]

        # power of mu for bias-corrected first and second moment estimate
        mu_power = tf.get_variable("mu_power", shape=(), dtype=tf.float32, trainable=False, initializer=tf.constant_initializer(1.0))
        # mu_power = tf.Variable(1.0, trainable=False)

        # save old mean and variance of grads
        old_ms = ms

        grads = tf.gradients(loss, var_list)
        grads_squared = [tf.square(g) for g in grads]

        m_updates = [m.assign(mu*m + (1.0-mu)*g) for m, g in zip(ms, grads)] #new means
        v_updates = [v.assign(mu*v + (1.0-mu)*g2) for v, g2 in zip(vs, grads_squared)]
        mu_power_update = [tf.assign(mu_power,tf.multiply(mu_power,mu))]

        #calculate probability of sign switch
        with tf.control_dependencies(v_updates+m_updates+mu_power_update):
            #bais_correction
            ms_hat = [tf.divide(m,tf.constant(1.0)- mu_power) for m in ms]
            vs_hat = [tf.divide(v,tf.constant(1.0) - mu_power) for v in vs]
            ms_squared = [tf.square(m) for m in ms_hat]
            rs = [tf.maximum(v-m2,tf.zeros_like(v)) for v, m2 in zip(vs_hat, ms_squared)] #new varience
            # probability of sign switch (with equal variance assumption)
            snrs = [tf.divide(m, tf.sqrt(r) + self._eps) for m, r in zip(grads, rs)]
            old_snrs = [tf.divide(m, tf.sqrt(r)+self._eps) for m, r in zip(old_grads,rs)]
            prob_greater_zero = [(0.5)*(1.0+tf.erf(tf.sqrt(1/2.0)*snr)) for snr in snrs]
            old_prob_greater_zero  =  [(0.5)*(1.0+tf.erf(tf.sqrt(1/2.0)*old_snr)) for old_snr in old_snrs]
            probs = [tf.multiply(p,1-old_p)+tf.multiply(old_p,1-p)
                     for p, old_p in zip(prob_greater_zero, old_prob_greater_zero)]
            # summary scalar switch_prob_0.75
            counds_prob=[tf.greater(prob, 0.75*tf.ones_like(prob)) for prob in probs]
            count_prob=[tf.reduce_sum(tf.cast(cond,tf.int64))
                        for cond in counds_prob]
            switch_prob = tf.add_n(count_prob)
            # summary histogram SNR
            abs_snrs =[tf.abs(snr) for snr in snrs]

            # find sign of product of gradients
            grads_sign = [tf.sign(grad) for grad in grads]
            old_grads_sign = [tf.sign(old_grad) for old_grad in old_grads]
            mult_sign_masks = [tf.multiply(grad_sign,old_grad_sign)
                               for grad_sign,old_grad_sign
                               in zip(grads_sign,old_grads_sign)]
            # check the product of signs
            conds_equal = [tf.equal(mult_sign_mask,
                                    tf.zeros_like(mult_sign_mask)) for mult_sign_mask in mult_sign_masks]
            conds_less = [tf.less(mult_sign_mask,
                                  tf.zeros_like(mult_sign_mask))
                          for mult_sign_mask in mult_sign_masks]
            conds_greater = [tf.greater(mult_sign_mask,tf.zeros_like(mult_sign_mask))
                             for mult_sign_mask in mult_sign_masks]

            # count the number of sign changes, the same signs and zero products
            # for every variable tensor
            # used for tracking of opt performance
            count_less=[tf.reduce_sum(tf.cast(cond_less,tf.int64))
                        for cond_less in conds_less]
            count_greater=[tf.reduce_sum(tf.cast(cond_greater,tf.int64))
                           for cond_greater in conds_greater]
            count_equal=[tf.reduce_sum(tf.cast(cond_equal,tf.int64))
                         for cond_equal in conds_equal ]
            # summary switch
            switch=tf.add_n(count_less)
            no_switch=tf.add_n(count_greater)
            zero_prod=tf.add_n(count_equal)
            n_of_parameters=switch+ no_switch + zero_prod
            sign_changes={"switch": switch/n_of_parameters,"no_switch":no_switch/n_of_parameters,
                          "zero_prod":zero_prod/n_of_parameters,"switch_prob": switch_prob/n_of_parameters}

            zeros=[tf.zeros_like(old_delta) for old_delta in old_deltas]


            # calculate the all possible delta updates
            deltas_equal=old_deltas
            deltas_less=[tf.maximum(d*eta_minus,
                                    tf.cast(tf.ones_like(d),tf.float32)*delta_min) for d in old_deltas]
            deltas_greater=[tf.minimum(d*eta_plus,tf.ones_like(d)*delta_max) for d in old_deltas]

            # select delta updates using cond tensors
            deltas=zeros
            deltas=[tf.where(cond_equal, delta_equal, delta) for (cond_equal,
                                                                  delta_equal,delta)
                        in zip(conds_equal,deltas_equal, deltas)]
            deltas=[tf.where(cond_less, delta_less, delta) for (cond_less, delta_less,
                                                                delta)
                        in zip(conds_less,deltas_less, deltas)]
            deltas=[tf.where(cond_greater, delta_greater, delta)
                    for (cond_greater, delta_greater, delta) in zip(conds_greater,
                                                                    deltas_greater, deltas)]

            # save new the deltas
            old_deltas_updates = [old_delta.assign(delta)
                                  for (old_delta, delta) in zip(old_deltas,deltas)]

            # calculate parameters update
            dirs_geq=[-delta*grad_sign
                      for  (delta,grad_sign) in zip(deltas,grads_sign)]

            # select no update in case of negative product
            # or dir_geq update in other cases
            dirs=zeros
            dirs=[tf.where(tf.logical_or(cond_equal,cond_greater), dir_geq, d)
                  for (cond_equal,cond_greater,dir_geq,d)
                  in zip(conds_equal,conds_greater,dirs_geq, dirs)]

            # change grad to zero in case of negative product and save new gradients
            grads_sign=[tf.where(cond_less,zero,grad_sign)
                        for (cond_less,zero,grad_sign) in zip(conds_less,zeros,grads_sign)]
            with tf.control_dependencies(probs+dirs):
                old_grads_updates = [old_grad.assign(g)
                                     for (old_grad, g) in zip(old_grads, grads)]

            # here learning rate is scaling parameter (in original paper there is no learning_rate)
            with tf.control_dependencies(old_grads_updates+old_deltas_updates):
                variable_updates = [v.assign_add(self._lr*d) for v, d in zip(var_list, dirs)]
            for_summaries ={"sign": sign_changes,"prob":probs,"snr":abs_snrs,"delta": deltas,"mu_power": mu_power}
        return tf.group(*variable_updates),for_summaries

class RPROPOptimizer(tf.train.GradientDescentOptimizer):

    def __init__(self, learning_rate, name="RPROP", delta_0=0.1,
                 delta_min=10 ^(-6), delta_max=50,
                 eta_minus=0.5, eta_plus=1.2):
        super(RPROPOptimizer, self).__init__(learning_rate, name=name)
        self._lr = learning_rate
        self._delta_0 = delta_0
        self._delta_min = delta_min
        self._delta_max = delta_max
        self._eta_minus = eta_minus
        self._eta_plus = eta_plus

    def minimize(self, loss, var_list=None):

        # Algo params as constant tensors
        delta_0=tf.convert_to_tensor(self._delta_0, dtype=tf.float32)
        delta_min=tf.convert_to_tensor(self._delta_min, dtype=tf.float32)
        delta_max=tf.convert_to_tensor(self._delta_max, dtype=tf.float32)
        eta_minus=tf.convert_to_tensor(self._eta_minus, dtype=tf.float32)
        eta_plus=tf.convert_to_tensor(self._eta_plus, dtype=tf.float32)

        if var_list is None:
            var_list = tf.trainable_variables()
            print(var_list)


        # Create and retrieve slot variables for delta , old_grad values
        # and old_dir (values of gradient changes)

        old_deltas = [self._get_or_make_slot(var,
                  tf.constant(0.1, tf.float32, var.get_shape()), "delta", "delta")
                  for var in var_list]
        old_grads_sign = [self._get_or_make_slot(var,
                  tf.constant(0, tf.float32, var.get_shape()), "grads_sign", "grads_sign")
                  for var in var_list]

        # old_dirs = [self._get_or_make_slot(var,
                #   tf.constant(0, tf.float32, var.get_shape()), "dir", "dir")
                #   for var in var_list]

        grads = tf.gradients(loss, var_list)
        grads_sign = [tf.sign(grad) for grad in grads]
        mult_sign_masks = [tf.multiply(grad_sign,old_grad_sign)
                           for grad_sign,old_grad_sign
                           in zip(grads_sign,old_grads_sign)]
        # check the product of signs
        conds_equal = [tf.equal(mult_sign_mask,
                                tf.zeros_like(mult_sign_mask)) for mult_sign_mask in mult_sign_masks]
        conds_less = [tf.less(mult_sign_mask,
                              tf.zeros_like(mult_sign_mask))
                      for mult_sign_mask in mult_sign_masks]
        conds_greater = [tf.greater(mult_sign_mask,tf.zeros_like(mult_sign_mask))
                         for mult_sign_mask in mult_sign_masks]

        # count the number of sign changes, the same signs and zero products
        # for every variable tensor
        # used for tracking of opt performance
        count_less=[tf.reduce_sum(tf.cast(cond_less,tf.int64))
                    for cond_less in conds_less]
        count_greater=[tf.reduce_sum(tf.cast(cond_greater,tf.int64))
                       for cond_greater in conds_greater]
        count_equal=[tf.reduce_sum(tf.cast(cond_equal,tf.int64))
                     for cond_equal in conds_equal ]
        # add results from all variables and save them as dict
        switch=tf.add_n(count_less)
        no_switch=tf.add_n(count_greater)
        zero_prod=tf.add_n(count_equal)

        sign_changes={"switch": switch,"no_switch":no_switch,
                      "zero_prod":zero_prod}

        zeros=[tf.zeros_like(old_delta) for old_delta in old_deltas]


        # calculate the all possible updates
        deltas_equal=old_deltas
        deltas_less=[tf.maximum(d*eta_minus,
                                tf.cast(tf.ones_like(d),tf.float32)*delta_min) for d in old_deltas]
        deltas_greater=[tf.minimum(d*eta_plus,tf.ones_like(d)*delta_max) for d in old_deltas]

        # select updates using cond tensors
        deltas=zeros
        deltas=[tf.where(cond_equal, delta_equal, delta) for (cond_equal,
                                                              delta_equal,delta)
                    in zip(conds_equal,deltas_equal, deltas)]
        deltas=[tf.where(cond_less, delta_less, delta) for (cond_less, delta_less,
                                                            delta)
                    in zip(conds_less,deltas_less, deltas)]
        deltas=[tf.where(cond_greater, delta_greater, delta)
                for (cond_greater, delta_greater, delta) in zip(conds_greater,
                                                                deltas_greater, deltas)]

        # save new the deltas
        old_deltas_updates = [old_delta.assign(delta)
                              for (old_delta, delta) in zip(old_deltas,deltas)]

        #
        dirs_geq=[-delta*grad_sign
                  for  (delta,grad_sign) in zip(deltas,grads_sign)]

        # we don't need it in the tech report implementation
        # dirs_less=[-old_dir
        #            for  old_dir in old_dirs]

        # select no update in case of negative product
        # or dir_geq update in other cases
        dirs=zeros
        dirs=[tf.where(tf.logical_or(cond_equal,cond_greater), dir_geq, d)
              for (cond_equal,cond_greater,dir_geq,d)
              in zip(conds_equal,conds_greater,dirs_geq, dirs)]

        # we don't need it in the tech report implementation
        # old_dirs_updates=[old_dir.assign(d) for old_dir,d in zip(old_dirs,dirs)]

        # change grad to zero in case of negative product and save new gradients
        grads_sign=[tf.where(cond_less,zero,grad_sign)
                    for (cond_less,zero,grad_sign) in zip(conds_less,zeros,grads_sign)]
        old_grads_updates = [old_grad.assign(g)
                             for (old_grad, g) in zip(old_grads_sign, grads_sign)]

        # here learning rate is scaling parameter (in original paper there is no learning_rate)
        # we need to check how quikly we change it during training
        with tf.control_dependencies(old_grads_updates+old_deltas_updates):
            variable_updates = [v.assign_add(self._lr*d) for v, d in zip(var_list, dirs)]
        return tf.group(*variable_updates),sign_changes
