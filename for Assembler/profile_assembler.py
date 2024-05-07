import pandas as pd
import numpy as np
from pathlib import Path
import sys, os
from ruamel.yaml import YAML
import importlib

cur_file = Path(__file__).resolve()
cur_folder = cur_file.parent
assembler_folder = cur_file.parents[3]
top_folder = cur_file.parents[4]
sys.path.append(str(top_folder))

algo_config_path = os.path.join(assembler_folder,'Blue Prints','sample_blue_prints.yaml')
algo_config_type = 'map'

# Set up the module function to be tested. This is equivalent to from X.Y.Z import module_func, but avoid potential ImportError
assembler_py_file_path = os.path.join(assembler_folder,'Assembler','AlgorithmAssembler.py')
name = os.path.relpath(assembler_py_file_path,top_folder).replace('.py','').replace(os.sep,'.')
spec = importlib.util.spec_from_file_location(name, assembler_py_file_path)
assembler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assembler)

# Prepare input
def prepare_input(assembler_folder,biolinq_file_name):
    algo_input = {}
    algo_input['biolinq'] = pd.read_csv(os.path.join(assembler_folder,'Test','UnitTests','Assembler','Input',biolinq_file_name))
    algo_input['mfg_data'] = {'sim_slope': 0.0001, 'post_cal_intercept': 0.02}
    return algo_input

def prepare_reference(assembler_folder,biolinq_file_name):
    biolinq_df = pd.read_csv(os.path.join(assembler_folder,'Test','UnitTests','Assembler','Reference',biolinq_file_name))
    return biolinq_df

def assess_gluc_max_abs_diff(biolinq_df,assembler_report_df):
    return assembler_report_df['biolinq']

# @profile
def test_equivalency_br232650020096():
    biolinq_df = prepare_reference(assembler_folder,'alg_br232650020096.csv')
    algo_input = prepare_input(assembler_folder,'biolinq_raw_br232650020096.csv')
    alg = assembler.AlgoAssembler()
    alg.assemble_algo(algo_config_path,algo_config_type)
    alg.reinitialize_algo()
    alg.run(algo_input)
    alg.dump_algo_output('customized')  
    alg.report_df.to_csv(os.path.join(assembler_folder,'Test','UnitTests','Assembler','Output','report_df_br232650020096.csv'), header=True)
    alg.sample_df.to_csv(os.path.join(assembler_folder,'Test','UnitTests','Assembler','Output','sample_df_br232650020096.csv'), header=True, float_format='%g')
    alg.dump_algo_output('commercial')
    alg.remove_algo_output_module_name_suffix()
    # v14h in Brandegee will resample and average ms_since_boot unintendedly, and thus it is not working to use ms_since_boot to merge.
    # For now just use index to merge two dataframes
    merged_df = pd.merge(biolinq_df, alg.report_df, how = 'left', left_index=True,right_index=True)
    # Test glucose is "aligned" with DataRunner v14h (MARD < 10) 
    assert merged_df['biolinq_x'].sub(merged_df['biolinq_y']).div(merged_df['biolinq_y']).abs().mean() < 0.15
    # Test glucose is "aligned" with DataRunner v14h (Corr < 0.9) 
    assert merged_df[['biolinq_x','biolinq_y']].corr().loc['biolinq_x','biolinq_y'] > 0.85
    alg.report_df.to_csv(os.path.join(assembler_folder,'Test','UnitTests','Assembler','Output','commercial_br232650020096.csv'), header=True)

def test_equivalency_br232570050088():
    biolinq_df = prepare_reference(assembler_folder,'alg_br232570050088.csv')
    algo_input = prepare_input(assembler_folder,'biolinq_raw_br232570050088.csv')
    alg = assembler.AlgoAssembler()
    alg.assemble_algo(algo_config_path,algo_config_type)
    alg.reinitialize_algo()
    alg.run(algo_input)
    alg.dump_algo_output('customized') 
    alg.report_df.to_csv(os.path.join(assembler_folder,'Test','UnitTests','Assembler','Output','report_df_br232570050088.csv'), header=True)
    alg.sample_df.to_csv(os.path.join(assembler_folder,'Test','UnitTests','Assembler','Output','sample_df_br232570050088.csv'), header=True, float_format='%g')    
    alg.dump_algo_output('commercial')
    alg.remove_algo_output_module_name_suffix()
    # v14h in Brandegee will resample and average ms_since_boot unintendedly, and thus it is not working to use ms_since_boot to merge.
    # For now just use index to merge two dataframes    
    merged_df = pd.merge(biolinq_df, alg.report_df, how = 'left', left_index=True,right_index=True)
    assert merged_df['biolinq_x'].sub(merged_df['biolinq_y']).div(merged_df['biolinq_y']).abs().mean() < 0.15
    # Test glucose is "aligned" with DataRunner v14h (Corr < 0.9) 
    assert merged_df[['biolinq_x','biolinq_y']].corr().loc['biolinq_x','biolinq_y'] > 0.85
    alg.report_df.to_csv(os.path.join(assembler_folder,'Test','UnitTests','Assembler','Output','commercial_br232570050088.csv'), header=True)
    
test_equivalency_br232650020096()