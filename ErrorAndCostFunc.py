from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error, median_absolute_error

def g_error_mse(params,sim,i,g,glucose_model):
    g_hat = glucose_model(sim,i,params)
    return mean_squared_error(g,g_hat)

def g_error_mard(params,sim,i,g,glucose_model):
    g_hat = glucose_model(sim,i,params)
    g_weight = g.rdiv(1)
    return mean_absolute_error(g,g_hat,sample_weight=g_weight)

def g_error_mard_weighted(params,sim,i,g,glucose_model):
    g_hat = glucose_model(sim,i,params)
    g_weight = g.rdiv(1)
    hypo_index = g[g<70].index
    hyper_index = g[g>180].index
    eu_index = g[(g>=70) & (g<=180)].index
    return 0.15*mean_absolute_error(g[hypo_index],g_hat[hypo_index],sample_weight=g_weight[hypo_index]) + \
            0.7*mean_absolute_error(g[hyper_index],g_hat[hyper_index],sample_weight=g_weight[hyper_index]) + \
            0.15*mean_absolute_error(g[eu_index],g_hat[eu_index],sample_weight=g_weight[eu_index])

def g_error_median_ard(params,sim,i,g,glucose_model):
    g_hat = glucose_model(sim,i,params)
    g_weight = g.rdiv(1)
    return median_absolute_error(g,g_hat,sample_weight=g_weight)

def cur_error_mse(params,sim,i,g,current_model):
    cur_true = current_model(sim,g,params)
    return mean_squared_error(cur_true,i)

def cur_error_mard(params,sim,i,g,current_model):
    cur_true = current_model(sim,g,params)
    cur_weight = g.rdiv(1)
    return mean_absolute_error(cur_true,i,sample_weight=cur_weight)

def cur_error_mard_weighted(params,sim,i,g,current_model):
    cur_true = current_model(sim,g,params)
    cur_weight = g.rdiv(1)   
    hypo_index = g[g<70].index
    hyper_index = g[g>180].index
    eu_index = g[(g>=70) & (g<=180)].index
    return 0.15*mean_absolute_error(cur_true[hypo_index],i[hypo_index],sample_weight=cur_weight[hypo_index]) + \
            0.7*mean_absolute_error(cur_true[hyper_index],i[hyper_index],sample_weight=cur_weight[hyper_index]) + \
            0.15*mean_absolute_error(cur_true[eu_index],i[eu_index],sample_weight=cur_weight[eu_index])

def cur_error_median_ard(params,sim,i,g,current_model):
    cur_true = current_model(sim,g,params)
    cur_weight = g.rdiv(1)
    return median_absolute_error(cur_true,i,sample_weight=cur_weight)
