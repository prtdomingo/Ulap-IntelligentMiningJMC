#%%
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import urllib.request

import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core.compute import AmlCompute, ComputeTarget



# check core SDK version number
print('Azure ML SDK Version: ', azureml.core.VERSION)

#%%
ws = Workspace.from_config()
print(ws.name, ws.location, ws.resource_group, sep='\t')
#%%
experiment_name = 'UlapExperiment'

from azureml.core import Experiment
exp = Experiment(workspace=ws, name=experiment_name)
#%%
compute_name = os.environ.get('AML_COMPUTE_CLUSTER_NAME', 'UlapCluster')
compute_min_nodes = os.environ.get('AML_COMPUTE_CLUSTER_MIN_NODES', 0)
compute_max_nodes = os.environ.get('AML_COMPUTE_CLUSTER_MAX_NODES', 4)

vm_size = os.environ.get('AML_COMPUTE_CLUSTER_SKU', 'STANDARD_D2_V2')

compute_target = None
provisioning_config = None

if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    
    if compute_target and type(compute_target) is AmlCompute:
            print('Found compute target. Just use it: ' + compute_name)
else:
    print('Creating a new compute target')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size, 
                                                                min_nodes = compute_min_nodes, 
                                                                max_nodes = compute_max_nodes)
    
    # Create The Cluster
    compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
    
    compute_target.wait_for_completion(show_output = True, min_node_count = True, timeout_in_minutes = 20)
    
    print(compute_target.status.serialize())
#%%
os.makedirs('./data', exist_ok = True)

# INSERT DATA SOURCE HERE
