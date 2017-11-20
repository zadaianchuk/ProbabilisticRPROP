from __future__ import division

import tensorflow as tf
import gradient_moment as gm

class ProbRPROPOptimizer(tf.train.GradientDescentOptimizer):

    def __init__(self, delta_0, learning_rate = 1, name="ProbRPROP", mu=0.95,
                 delta_min=10**(-9), delta_max=0.05,
                 eta_minus=0.5, eta_plus=1.2, p_min=0.75, eps=1e-8):
        super(ProbRPROPOptimizer, self).__init__(learning_rate, name=name)

        self._mu = mu
        self._lr = learning_rate
        self._delta_0 = delta_0
        self._delta_min = delta_min
        self._delta_max = delta_max
        self._eta_minus = eta_minus
        self._eta_plus = eta_plus
        self._eps=eps
        self._p_min = p_min

    def minimize(self, losses, global_step, var_list=None, USE_MINIBATCH_ESTIMATE = True, MAKE_NEG_STEP = True):

        # Algo params as constant tensors
        mu = tf.convert_to_tensor(self._mu, dtype=tf.float32)
        delta_0=tf.convert_to_tensor(self._delta_0, dtype=tf.float32)
        delta_min=tf.convert_to_tensor(self._delta_min, dtype=tf.float32)
        delta_max=tf.convert_to_tensor(self._delta_max, dtype=tf.float32)
        eta_minus=tf.convert_to_tensor(self._eta_minus, dtype=tf.float32)
        eta_plus=tf.convert_to_tensor(self._eta_plus, dtype=tf.float32)
        p_min = self._p_min
        if var_list is None:
            var_list = tf.trainable_variables()
            print(var_list)


        # Create and retrieve slot variables for delta , old_grad values
        # and old_dir (values of gradient changes)

        old_deltas = [self._get_or_make_slot(var,
                  tf.constant(self._delta_0, tf.float32, var.get_shape()), "delta", "delta")
                  for var in var_list]
        old_grads = [self._get_or_make_slot(var,
                  tf.constant(0, tf.float32, var.get_shape()), "grads", "grads")
                  for var in var_list]

        if not USE_MINIBATCH_ESTIMATE:
            loss = tf.reduce_mean(losses, name = "loss" )
            # moving average estimation
            ms = [self._get_or_make_slot(var,
                tf.constant(0.0, tf.float32, var.get_shape()), "m", "m")
                for var in var_list]
            vs = [self._get_or_make_slot(var,
                tf.constant(0.0, tf.float32, var.get_shape()), "v", "v")
                for var in var_list]
            # power of mu for bias-corrected first and second moment estimate
            mu_power = tf.get_variable("mu_power", shape=(), dtype=tf.float32, trainable=False, initializer=tf.constant_initializer(1.0))


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
        else:
            grads, grad_moms, loss, batch_size = gm.grads_and_grad_moms(losses, var_list)
            grads_squared =  [tf.square(grad) for grad in grads]
            rs = [tf.maximum(tf.divide(m-g2, batch_size-1.0), 0.0) for g2, m in zip(grads_squared, grad_moms)]


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
        prods = [tf.multiply(grad_sign,old_grad_sign)
                           for grad_sign,old_grad_sign
                           in zip(grads_sign,old_grads_sign)]
        with tf.control_dependencies(prods+probs):
            # check the product of signs
            conds_equal = [tf.equal(prod,tf.zeros_like(prod)) for prod in prods]
            conds_less = [tf.less(prod,tf.zeros_like(prod)) for prod in prods]
            conds_greater = [tf.greater(prod,tf.zeros_like(prod)) for prod in prods]

            # count the number of sign changes, the same signs and zero products
            # for every variable tensor
            # used for tracking of opt performance
            count_less=[tf.reduce_sum(tf.cast(cond_less,tf.int64))
                        for cond_less in conds_less]
            count_greater=[tf.reduce_sum(tf.cast(cond_greater,tf.int64))
                           for cond_greater in conds_greater]
            count_equal=[tf.reduce_sum(tf.cast(cond_equal,tf.int64))
                         for cond_equal in conds_equal ]

            probs_between = [tf.logical_and(tf.greater(prob,p_min*tf.ones_like(prob)),tf.less(prob,(1-p_min)*tf.ones_like(prob))) for prob in probs]
            probs_near_zero = [tf.less(prob,p_min*tf.ones_like(prob)) for prob in probs]
            probs_near_one = [tf.greater(prob,(1-p_min)*tf.ones_like(prob)) for prob in probs]


            # summary switch
            switch=tf.add_n(count_less)
            no_switch=tf.add_n(count_greater)
            zero_prod=tf.add_n(count_equal)
            n_of_parameters=switch+ no_switch + zero_prod
            sign_changes={"switch": switch/n_of_parameters,"no_switch":no_switch/n_of_parameters,
                          "zero_prod":zero_prod/n_of_parameters,"switch_prob": switch_prob/n_of_parameters}

            zeros=[tf.zeros_like(old_delta) for old_delta in old_deltas]
            # calculate the all possible delta updates
            deltas_less=[tf.maximum(d*eta_minus,
                                    tf.cast(tf.ones_like(d),tf.float32)*delta_min) for d in old_deltas]
            deltas_greater=[tf.minimum(d*eta_plus,tf.ones_like(d)*delta_max) for d in old_deltas]

            # select delta updates using cond tensors
            deltas=old_deltas
            deltas=[tf.where(cond_less, delta_less, delta) for (cond_less, delta_less,
                                                                delta)
                        in zip(probs_near_one,deltas_less, deltas)]
            deltas=[tf.where(cond_greater, delta_greater, delta)
                    for (cond_greater, delta_greater, delta) in zip(probs_near_zero,
                                                                    deltas_greater, deltas)]

            delta_mins = [tf.equal(delta,tf.cast(tf.ones_like(delta),tf.float32)*delta_min) for delta in deltas]
            count_min=[tf.reduce_sum(tf.cast(delta_min,tf.int64))
                        for delta_min in delta_mins]
            delta_maxs = [tf.equal(delta,tf.cast(tf.ones_like(delta),tf.float32)*delta_max) for delta in deltas]
            count_max=[tf.reduce_sum(tf.cast(delta_max,tf.int64))
                        for delta_max in delta_maxs]
            delta_min_count=tf.add_n(count_min)/n_of_parameters
            delta_max_count=tf.add_n(count_max)/n_of_parameters
            # save new the deltas
            old_deltas_updates = [old_delta.assign(delta)
                                  for (old_delta, delta) in zip(old_deltas,deltas)]

            # w update directions
            # select no update in case of negative product
            # or dir_geq update in other cases
            with tf.control_dependencies(old_deltas_updates):
                if MAKE_NEG_STEP:
                    dirs = [-delta*grad_sign
                              for  (delta,grad_sign) in zip(deltas,grads_sign)]
                else:
                    dirs=zeros
                    dirs_geq=[-delta*grad_sign
                              for  (delta,grad_sign) in zip(deltas,grads_sign)]
                    dirs=[tf.where(tf.logical_or(cond_equal,cond_greater), dir_geq, d)
                          for (cond_equal,cond_greater,dir_geq,d)
                          in zip(probs_between,prob_near_zero,dirs_geq, dirs)]

                # change grad to zero in case of negative product and save new gradients
                # grads=[tf.where(cond_less,zero,grad)
                #             for (cond_less,zero,grad) in zip(probs_near_one,zeros,grads)]
                old_grads_updates = [old_grad.assign(g)
                                     for (old_grad, g) in zip(old_grads, grads)]

                with tf.control_dependencies(dirs):
                    old_grads_updates = [old_grad.assign(g)
                                         for (old_grad, g) in zip(old_grads, grads)]

                    # here learning rate is scaling parameter, by default equal 1
                    with tf.control_dependencies(old_grads_updates):
                        variable_updates = [v.assign_add(self._lr*d) for v, d in zip(var_list, dirs)]
                        for_summaries ={"sign": sign_changes,"prob":probs,"snr":abs_snrs,"delta": deltas}
                        global_step.assign_add(1)
                        with tf.name_scope("summaries"):
                            with tf.name_scope("per_iteration"):
                                min_sum = tf.summary.scalar("min", delta_min_count, collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
                                max_sum = tf.summary.scalar("max", delta_max_count, collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
                                switch_sum = tf.summary.scalar("switch", sign_changes["switch"], collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
                                switch_prob_sum = tf.summary.scalar("switch_prob>0.75",sign_changes["switch_prob"], collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
                                for (i,prob) in enumerate(probs):
                                    prob_sum = tf.summary.histogram("switch_hist/"+str(i), prob, collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
                                for (i,snr) in enumerate(abs_snrs):
                                    snr_sum = tf.summary.histogram("snr_hist/"+str(i), snr, collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
                                for (i,delta) in enumerate(deltas):
                                    #find and clip big values
                                    cond =tf.greater(delta,tf.contrib.distributions.percentile(delta,85.0)*tf.ones_like(delta))
                                    delta2 = tf.where(cond, tf.contrib.distributions.percentile(delta,85.0)*tf.ones_like(delta), delta)
                                    delta_sum = tf.summary.histogram("delta_hist/"+str(i), delta2, collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
            return tf.group(*variable_updates)
