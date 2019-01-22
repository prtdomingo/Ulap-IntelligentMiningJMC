#%%
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import urllib.request
import logging

import azureml.core
from azureml.core import Workspace, Experiment, Run, Datastore
from azureml.core.compute import AmlCompute, ComputeTarget
import azureml.dataprep as dprep
from azureml.train.estimator import Estimator
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun
from sklearn.model_selection import train_test_split



# check core SDK version number
print('Azure ML SDK Version: ', azureml.core.VERSION)

#%%
ws = Workspace.from_config()
print(ws.name, ws.location, ws.resource_group, sep='\t')
#%%
project_folder = './Ulap-IntelligentMiningJMC'
experiment_name = 'UlapExperiment'

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
<<<<<<< HEAD
os.makedirs('./data', exist_ok = True)

# INSERT DATA SOURCE HERE
=======
ds = ws.get_default_datastore()
print(ds.datastore_type, ds.account_name, ds.container_name)
# ds.upload(src_dir='./data', target_path='AssetData', overwrite=True, show_progress=True)

#%%
asset_data_path = 'data/AssetData_Historical.csv'
asset_data_df = dprep.read_csv(path=asset_data_path, header=dprep.PromoteHeadersMode.GROUPED)
display(asset_data_df.head(5))
#%%
dprep_path = os.path.join(os.getcwd(), 'dflows.dprep')
dflow_prepared = asset_data_df
package = dprep.Package([dflow_prepared])
package.save(dprep_path)
#%%
package_saved = dprep.Package.open(dprep_path)
dflow_prepared = package_saved.dataflows[0]
dflow_prepared.get_profile()

#%%
int_type = dprep.TypeConverter(dprep.FieldType.INTEGER)
dflow_prepared = dflow_prepared.set_column_types(type_conversions={
    'Failure_NextHour': int_type
})


#%%
dflow_X = dflow_prepared.keep_columns(['Density_Overload', 'Abnormal_Flow_Rate', 
'Heat_Flow', 'Asset_Integrity', 'Temperature_Differential', 
'Volumetric_Flow_Rate', 'Tangential_Stress', 'Duct_Lenghts_in_Units', 
'Fault_in_last_Month', 'Avg_hours_in_Use', 'Pressure_Alarm',
'Inclination_Angle', 'Operating_Pressure_above_Normal', 'Compression_Ratio',
'Multiple_Connects','Water_Exposure_units', 'Humidity_Factor',
'Cathodic_Protection', 'Pressure_Class', 'District',
'Latitude, Longitude'])

dflow_y = dflow_prepared.keep_columns('Failure_NextHour')

x_df = dflow_X.to_pandas_dataframe()
y_df = dflow_y.to_pandas_dataframe()

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=223)
y_train.values.flatten()

#%%
automl_settings = {
    "iteration_timeout_minutes" : 10,
    "iterations" : 30,
    "primary_metric" : 'spearman_correlation',
    "preprocess" : True,
    "verbosity" : logging.INFO,
    "n_cross_validations": 5
}

automated_ml_config = AutoMLConfig(task = 'regression',
                             debug_log = 'automated_ml_errors.log',
                             path = project_folder,
                             X = x_train.values,
                             y = y_train.values.flatten(),
                             **automl_settings)

local_run = exp.submit(automated_ml_config, show_output=True)
>>>>>>> 17f911018c15f365fd707db22f45197f994f16ec
