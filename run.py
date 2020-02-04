import logging as log
import numpy as np
import pprint as pp
import pyNN.neuroml as sim

class RunSim(object):

    def __init__(self, name, params):
        self._network = None
        self.name = name

    def run(self, params):
        self._do_record = False
        self._network = {}
        self._network['timestep'] = 0.1 #ms
        self._network['min_delay'] = 0.1 #ms
        self._network['run_time'] = 1000
        
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
        log.info('Running sim for {}ms.'.format(self._network['run_time']))
        sim.run(self._network['run_time'])

        # Collect data
        if self._do_record:
            log.info('Collect data from all populations:')
            for popname, pop in pops.items():
                log.info(' -> Saving recorded data for "{}".'.format(popname))
                # Voltages
                voltagedata = pop.get_data('v')
                signal = voltagedata.segments[0].analogsignals[0]
                source_ids = signal.annotations['source_ids']
                for idx in range(len(source_ids)):
                    s_id = source_ids[idx]
                    filename = "%s_%s_%s.dat"%(pop.label,pop.id_to_index(s_id),signal.name)
                    vm = signal.transpose()[idx]
                    tt = np.array([t*sim.get_time_step()/1000. for t in range(len(vm))])
                    times_vm = np.array([tt, vm/1000.]).transpose()
                    np.savetxt(filename, times_vm, delimiter = '\t', fmt='%s')
                
                    # Spikes
                    spikedata = pop.get_data('spikes')
                    filename = '{}.spikes'.format(pop.label)
                    thefile = open(filename, 'w')
                    for spiketrain in spikedata.segments[0].spiketrains:
                        source_id = spiketrain.annotations['source_id']
                        source_index = spiketrain.annotations['source_index']
                        #log.info(pp.pprint(vars(spiketrain)))
                        for t in spiketrain:
                            thefile.write('%s\t%f\n'%(source_index,t.magnitude/1000.))
                            thefile.close()
            
        # End
        sim.end()

    def gen_input_pop(self, params=None):
        params = {
            'cm': 0.09,  # nF
            'v_reset': -70.,  # mV
            'v_rest': -65.,  # mV
            'v_thresh': -55.0,  # mV
            'tau_m': 10.,  # ms
            'tau_refrac': 1.,  # ms
            'tau_syn_E': 1., # ms
            'tau_syn_I': 1., # ms
        }
        pop = sim.Population(5, # Num. neurons in pop.
                             sim.IF_curr_exp, # Neuron type
                             params,
                             label='input')
        if self._do_record:
            pop.record('v')
            pop.record('spikes')
        return pop

    def gen_input_to_output_proj(self, params=None):
        pops = self._network['populations']
        proj = sim.Projection(pops['input'],
                              pops['output'],
                              sim.AllToAllConnector(),
                              sim.StaticSynapse(weight=-5.0,
                                                delay=0.1, # ms (equals timestep here)
                                                #label='input to output',
                                                #receptor_type='inhibitory'
                              ))
        return proj
                                   

    def gen_output_pop(self, params=None):
        params = {
            'cm': 0.09,  # nF
            'v_reset': -70.,  # mV
            'v_rest': -65.,  # mV
            'v_thresh': -55.0,  # mV
            'tau_m': 10.,  # ms
            'tau_refrac': 1.,  # ms
            'tau_syn_E': 1., # ms
            'tau_syn_I': 1., # ms
        }
        pop = sim.Population(5,
                             sim.IF_curr_exp,
                             params,
                             label='output')
        if self._do_record:
            pop.record('v')
            pop.record('spikes')
        return pop

r = RunSim('Runner', {})
r.run({})
