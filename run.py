import pynn_genn as sim

class RunSim(object):

    def __init__(self, name, params):
        self._network = None
        self.name = name

    def run(self, params):
        self._network = {}
        self._network['timestep'] = 0.1 #ms
        self._network['min_delay'] = 0.1 #ms
        self._network['run_time'] = 10
        
        sim.setup(self._network['timestep'],
                  model_name=self.name,
                  backend='CUDA')

        # Neuron populations
        pops = {}
        pops['input'] = self.gen_input_pop()
        pops['output'] = self.gen_output_pop()
        self._network['populations'] = pops

        # Projections
        projs = {}
        projs['inupt to output'] = self.gen_input_to_output_proj()
        self._network['projections'] = projs

        # Run
        sim.run(self._network['run_time'])
        sim.end()

    def gen_input_pop(self, params=None):
        params = {
            'cm': 0.09,  # nF
            'v_reset': -70.,  # mV
            'v_rest': -65.,  # mV
            'v_thresh': VTHRESH,  # mV
            'tau_m': 10.,  # ms
            'tau_refrac': 1.,  # ms
            'tau_syn_E': 1., # ms
            'tau_syn_I': 1., # ms
        }
        pop = sim.Population(5, # Num. neurons in pop.
                             'IF_curr_exp', # Neuron type
                             params,
                             label='input')
        return pop

    def gen_input_to_output_proj(self, params=None):
        proj = sim.Projection(pops['input'],
                              pops['output'],
                              sim.AllToAllConnector(),
                              sim.StaticSynapse(weight=iw,
                                                delay=0.1, # ms (equals timestep here)
                                                label='input to output',
                                                receptor_type='inhibitory'))
        return proj
                                   

    def gen_output_pop(self, params=None):
        params = {
            'cm': 0.09,  # nF
            'v_reset': -70.,  # mV
            'v_rest': -65.,  # mV
            'v_thresh': VTHRESH,  # mV
            'tau_m': 10.,  # ms
            'tau_refrac': 1.,  # ms
            'tau_syn_E': 1., # ms
            'tau_syn_I': 1., # ms
        }
        pop = sim.Population(5,
                             'IF_curr_exp',
                             params,
                             label='output')
        return pop
