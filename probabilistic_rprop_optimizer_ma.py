from __future__ import division

import tensorflow as tf
import gradient_moment as gm



class ProbRPROPOptimizer(tf.train.GradientDescentOptimizer):

    def __init__(self, delta_0, learning_rate = 1, name="ProbRPROP", mu=0.99,
                 delta_min=10**(-9), delta_max=0.05,
                 eta_minus=0.5, eta_plus=1.2, p_min=0.75, eps=1e-8, eta_type ="exponential"):
        super(ProbRPROPOptimizer, self).__init__(learning_rate, name=name)

        self._mu = mu
        self._delta_0 = delta_0
        self._delta_min = delta_min
        self._delta_max = delta_max
        self._eta_minus = eta_minus
        self._eta_plus = eta_plus
        self._eps=eps
        self._p_min = p_min
        self.eta_type = eta_type

    def _calculate_eta(self, prob):
        eta_type = self.eta_type
        eta_minus = tf.convert_to_tensor(self._eta_minus, dtype=tf.float32)
        eta_plus = tf.convert_to_tensor(self._eta_plus, dtype=tf.float32)
        if eta_type == "interval":
            p_min = self._p_min
            eta = tf.ones_like(prob)
            prob_near_zero = tf.less(prob,p_min*tf.ones_like(prob))
            prob_near_one = tf.greater(prob,(1-p_min)*tf.ones_like(prob))
            eta = tf.where(prob_near_one , eta_minus*tf.ones_like(prob), eta)
            eta = tf.where(prob_near_zero, eta_plus*tf.ones_like(prob), eta)
        if eta_type == "linear":
            eta = eta_plus*(1-prob)+prob*eta_minus
        if eta_type == "exponential":
            eta = tf.exp(-2*tf.log(eta_plus)*(prob-0.5))
        return eta


    def minimize(self, loss, global_step, var_list=None, SOFT_SIGN = False):

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
        old_probs_greater_zero = [self._get_or_make_slot(var,
                  tf.constant(0.5, tf.float32, var.get_shape()), "prob_g_z", "prob_g_z")
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
        snrs_for_d = [tf.divide(m, tf.sqrt(r) + self._eps) for m, r in zip(ms_hat, rs)]
        probs_greater_zero_for_d = [(0.5)*(1.0+tf.erf(tf.sqrt(1/2.0)*snr)) for snr in snrs_for_d]
        probs_greater_zero = [(0.5)*(1.0+tf.erf(tf.sqrt(1/2.0)*snr)) for snr in snrs]
        probs = [tf.multiply(p,1-old_p)+tf.multiply(old_p,1-p)
                 for p, old_p in zip(probs_greater_zero, old_probs_greater_zero)]

        # summary histogram SNR
        abs_snrs =[tf.abs(snr) for snr in snrs]

        # find sign of product of gradients
        grads_sign = [tf.sign(grad) for grad in grads]
        with tf.control_dependencies(probs+grads_sign):

            etas = [self._calculate_eta(prob) for prob in probs]
            deltas=[tf.multiply(eta,delta) for (eta,delta) in zip(etas,old_deltas)]
            # calculate the all possible delta updates
            deltas = [tf.maximum(d, tf.ones_like(d)*delta_min) for d in deltas]
            deltas = [tf.minimum(d,tf.ones_like(d)*delta_max) for d in deltas]
            n_of_parameters = tf.add_n([tf.reduce_sum(tf.cast(tf.ones_like(delta),tf.int64))
                        for delta in deltas])

            delta_mins = [tf.equal(delta,tf.cast(tf.ones_like(delta),tf.float32)*delta_min) for delta in deltas]
            count_min=[tf.reduce_sum(tf.cast(delta_min,tf.int64))
                        for delta_min in delta_mins]
            delta_maxs = [tf.equal(delta,tf.cast(tf.ones_like(delta),tf.float32)*delta_max) for delta in deltas]
            count_max=[tf.reduce_sum(tf.cast(delta_max,tf.int64))
                        for delta_max in delta_maxs]
            delta_min_count=tf.add_n(count_min)
            delta_max_count=tf.add_n(count_max)
            # save new the deltas
            old_deltas_updates = [old_delta.assign(delta)
                                  for (old_delta, delta) in zip(old_deltas,deltas)]

            # w update directions
            # select no update in case of negative product
            # or dir_geq update in other cases
            with tf.control_dependencies(old_deltas_updates):
                if SOFT_SIGN:
                    ds = [2*p-1 for p in probs_greater_zero_for_d]
                else:
                    ds = grads_sign
                dirs = [-delta*d
                          for  (delta,d) in zip(deltas,ds)]
                # else:
                #     dirs=zeros
                #     dirs_geq=[-delta*d
                #               for  (delta,d) in zip(deltas,ds)]
                #     dirs=[tf.where(tf.logical_or(cond_equal,cond_greater), dir_geq, d)
                #           for (cond_equal,cond_greater,dir_geq,d)
                #           in zip(probs_between,prob_near_zero,dirs_geq, dirs)]

                with tf.control_dependencies(dirs):
                    old_probs_greater_zero_updates = [old_prob.assign(prob)
                                         for (old_prob, prob) in zip(old_probs_greater_zero, probs_greater_zero)]

                    # here learning rate is scaling parameter, by default equal 1
                    with tf.control_dependencies(old_probs_greater_zero_updates):
                        variable_updates = [v.assign_add(d) for v, d in zip(var_list, dirs)]
                        global_step.assign_add(1)
                        with tf.name_scope("summaries"):
                            with tf.name_scope("per_iteration"):
                                min_sum = tf.summary.scalar("min", delta_min_count, collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
                                max_sum = tf.summary.scalar("max", delta_max_count, collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
                                for (i,prob) in enumerate(probs):
                                    prob_sum = tf.summary.histogram("switch_hist/"+str(i), prob, collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
                                for (i,prob) in enumerate(probs_greater_zero):
                                    prob_sum = tf.summary.histogram("prob_zero_hist/"+str(i), prob, collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
                                for (i,snr) in enumerate(abs_snrs):
                                    snr_sum = tf.summary.histogram("snr_hist/"+str(i), snr, collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
                                for (i,delta) in enumerate(deltas):
                                    #find and clip big values
                                    cond =tf.greater(delta,tf.contrib.distributions.percentile(delta,85.0)*tf.ones_like(delta))
                                    delta2 = tf.where(cond, tf.contrib.distributions.percentile(delta,85.0)*tf.ones_like(delta), delta)
                                    delta_sum = tf.summary.histogram("delta_hist/"+str(i), delta2, collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
            return tf.group(*variable_updates)
