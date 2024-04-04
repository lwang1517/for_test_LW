import numpy as np

class GlucoseModel:
    def __init__(self):
        self.sim_slope_bounds = [[1e-3,2e3]]
        
        self.sim_slope_constant_background_bounds = [[1e-3,2e3],[-1,1]]
        self.sim_slope_constant_compartment_bounds = [[1e-3,2e3],[-100,100]]
        self.sim_slope_linear_background_bounds = [[1e-3,2e3],[1e-6,2e6],[-1,1]]
        self.sim_slope_nonlinear_background_bounds = [[1e-3,2e3],[1e-4,1],[1e-4,1e8]]

        self.sim_slope_intercept_constant_background_bounds = [[1e-3,2e3],[-10,10],[-1,1]]
        self.sim_slope_intercept_constant_compartment_bounds = [[1e-3,2e3],[-10,10],[-100,100]]
        self.sim_slope_intercept_linear_background_bounds = [[1e-3,2e3],[-10,10],[1e-6,2e6],[-1,1]]
        self.sim_slope_intercept_nonlinear_background_bounds = [[1e-3,2e3],[-10,10],[1e-4,1],[1e-4,1e8]]

        self.fmm_constant_background_bounds = [[1e-10,1e6],[1e-10,1e6],[1e-3,3e4],[-1,1]]
        self.fmm_constant_compartment_bounds = [[1e-4,1e5],[1e-4,1e5],[1e-3,3e4],[-100,100]]
        self.fmm_linear_background_bounds = [[1e-4,1e5],[1e-4,1e5],[1e-3,3e4],[1e-6,2e6],[-1,1]]
        self.fmm_nonlinear_background_bounds = [[1e-4,1e5],[1e-4,1e5],[1e-3,3e4],[1e-4,1],[1e-4,1e8]]              

    # sensitivity model sim_slope: gluc_current = g * 1/mag * sim_slope
    @staticmethod 
    def sim_slope(sim,gluc_cur,sim_slope):
        # output is glucose concentration
        return gluc_cur.mul(sim).div(sim_slope)

    @staticmethod 
    def sim_slope_inv(sim,g,sim_slope):
        # output is glucose current
        return g.div(sim).mul(sim_slope)

    # sensitivity model sim_slope_intercept: gluc_current = g * (1/mag * sim_slope + sim_intercept)
    @staticmethod 
    def sim_slope_intercept(sim,gluc_cur,sim_slope,sim_intercept):
        # output is glucose concentration
        sensitivity = sim.rdiv(1).mul(sim_slope).add(sim_intercept)
        return gluc_cur.div(sensitivity)

    @staticmethod 
    def sim_slope_intercept_inv(sim,g,sim_slope,sim_intercept):
        # output is glucose current
        sensitivity = sim.rdiv(1).mul(sim_slope).add(sim_intercept)
        return g.mul(sensitivity)
    
    # sensitivity model fmm: fick's law + Michaelis Menten
    # roc sg(t) = D1*(bg(t)-sg(t)) – K*cur(t)
    # K*cur = Vmax*sg(t)/(Km+sg(t))
    # We can assume roc sg(t) = 0 for steady state
    # D = 1/mag * sim slope

    @staticmethod 
    def fmm_constant(sim,sim_slope):
        faraday_constant = 9.64853321233100184e4 # C/mole
        nanoamp_to_coulomb_per_sec = 1e-9
        tip_h = 80 # micrometer
        tip_r = 70 # micrometer
        l_enzyme = 60 # micrometer
        l_pt = 60 #micrometer
        glm_thickness = 0.3 # micrometer
        glucose_mg_per_mole = 1.8e5
        um3_to_dl = 1e-14
        k0 = nanoamp_to_coulomb_per_sec/faraday_constant
        l = np.sqrt(tip_h**2+tip_r**2)
        l2 = l-(l_enzyme-l_pt)/2
        r2 = tip_r * l2/l
        S2 = np.pi*r2*l2
        l1 = l-((l_enzyme-l_pt)/2+l_pt)
        r1 = tip_r * l1/l
        S1 = np.pi*r1*l1
        S_pt = S2-S1
        V = S_pt * glm_thickness * um3_to_dl
        k1 = k0 * glucose_mg_per_mole / (2*V)
        D = sim.rdiv(1).mul(sim_slope)
        return D,k1
    
    def fmm(self,sim,gluc_cur,sim_slope,vmax,km):
        # output is glucose concentration
        D,k1 = self.fmm_constant(sim,sim_slope)
        gluc_consumed = gluc_cur.mul(k1)
        g_sensor_numerator = gluc_consumed.mul(km)
        g_sensor_denominator = gluc_consumed.mul(-1).add(vmax)
        g_sensor = g_sensor_numerator.div(g_sensor_denominator)
        g = g_sensor + gluc_consumed.div(D)
        return g

    def fmm_inv(self,sim,g,sim_slope,vmax,km):
        # output is glucose current
        D,k1 = self.fmm_constant(sim,sim_slope)
        a = 1
        b = -(vmax+D*km+D*g)
        c = vmax*D*g
        return (-b-np.sqrt(b**2-4*a*c))/2/k1

    # background model constant_background: gluc_current = i - background
    @staticmethod
    def constant_background(i,background):
        # output glucose current
        return i.sub(background)
    
    @staticmethod
    def constant_background_inv(gluc_cur,background):
        # output current
        return gluc_cur.add(background)
    
    # background model constant_compartment: current = (ref - compartment) * sensitivity
    @staticmethod
    def constant_compartment(dg,compartment):
        # output blood glucose from dermis glucose
        return dg.add(compartment)
    
    @staticmethod
    def constant_compartment_inv(g,compartment):
        # output dermis glucose from blood glucose
        return g.sub(compartment)

    # background model linear_background: gluc_current = i - (1/mag * background_slope + background_intercept)
    @staticmethod
    def linear_background(sim,i,background_slope,background_intercept):
        # output glucose current
        background = sim.rdiv(1).mul(background_slope).add(background_intercept)      
        return i.sub(background)
    
    @staticmethod
    def linear_background_inv(sim,gluc_cur,background_slope,background_intercept):
        # output current
        background = sim.rdiv(1).mul(background_slope).add(background_intercept) 
        return gluc_cur.add(background)

    # background model nonlinear_background: gluc_current = i - 1/(1/a0+mag/a1)
    @staticmethod
    def nonlinear_background(sim,i,background_a0,background_a1):
        # output glucose current
        background = sim.div(background_a1).add(1/background_a0).rdiv(1)    
        return i.sub(background)
    
    @staticmethod
    def nonlinear_background_inv(sim,gluc_cur,background_a0,background_a1):
        # output current
        background = sim.div(background_a1).add(1/background_a0).rdiv(1)
        return gluc_cur.add(background)

    ############## model based on sim_slope sensitivity model #############

    # Model0: current = ref * 1/mag * sim_slope
    def sim_slope_gluc(self,sim,i,params):
        return self.sim_slope(sim,i,params[0])

    def sim_slope_cur(self,sim,g,params):
        return self.sim_slope_inv(sim,g,params[0])
    
    # Model1: current = ref * 1/mag * sim_slope + background
    def sim_slope_constant_background_gluc(self,sim,i,params):
        sim_slope = params[0]
        background = params[1]
        gluc_cur = self.constant_background(i,background)
        g = self.sim_slope(sim,gluc_cur,sim_slope)
        return g

    def sim_slope_constant_background_cur(self,sim,g,params):
        sim_slope = params[0]
        background = params[1]
        gluc_cur = self.sim_slope_inv(sim,g,sim_slope)
        i = self.constant_background_inv(gluc_cur,background)
        return i
    
    # Model2: current = (ref - compartment) * 1/mag * sim_slope
    def sim_slope_constant_compartment_gluc(self,sim,i,params):
        sim_slope = params[0]
        compartment = params[1]
        dg = self.sim_slope(sim,i,sim_slope)
        g = self.constant_compartment(dg,compartment)
        return g
    
    def sim_slope_constant_compartment_cur(self,sim,g,params):
        sim_slope = params[0]
        compartment = params[1]
        dg = self.constant_compartment_inv(g,compartment)
        i = self.sim_slope_inv(sim,dg,sim_slope)
        return i
    
    # Model3: current = ref * 1/mag * sim_slope + (1/mag * background_slope + background_intercept)
    def sim_slope_linear_background_gluc(self,sim,i,params):
        sim_slope = params[0]
        background_slope = params[1]
        background_intercept = params[2]
        gluc_cur = self.linear_background(sim,i,background_slope,background_intercept)
        g = self.sim_slope(sim,gluc_cur,sim_slope)
        return g
    
    def sim_slope_linear_background_cur(self,sim,g,params):
        sim_slope = params[0]
        background_slope = params[1]
        background_intercept = params[2]
        gluc_cur = self.sim_slope_inv(sim,g,sim_slope)
        i = self.linear_background_inv(sim,gluc_cur,background_slope,background_intercept)
        return i
    
    # Model4: current = ref * 1/mag * sim_slope + (1/(1/a0+mag/a1))
    def sim_slope_nonlinear_background_gluc(self,sim,i,params):
        sim_slope = params[0]
        background_a0 = params[1]
        background_a1 = params[2]
        gluc_cur = self.nonlinear_background(sim,i,background_a0,background_a1)
        g = self.sim_slope(sim,gluc_cur,sim_slope)
        return g
    
    def sim_slope_nonlinear_background_cur(self,sim,g,params):
        sim_slope = params[0]
        background_a0 = params[1]
        background_a1 = params[2]
        gluc_cur = self.sim_slope_inv(sim,g,sim_slope)
        i = self.nonlinear_background_inv(sim,gluc_cur,background_a0,background_a1)
        return i
    
    ########## model based on sim_slope_intercept sensitivity model ######## 

    # Model5: current = ref * (1/mag * sim_slope +sim_intercept) + background
    def sim_slope_intercept_constant_background_gluc(self,sim,i,params):
        sim_slope = params[0]
        sim_intercept = params[1]
        background = params[2]
        gluc_cur = self.constant_background(i,background)
        g = self.sim_slope_intercept(sim,gluc_cur,sim_slope,sim_intercept)
        return g

    def sim_slope_intercept_constant_background_cur(self,sim,g,params):
        sim_slope = params[0]
        sim_intercept = params[1]
        background = params[2]
        gluc_cur = self.sim_slope_intercept_inv(sim,g,sim_slope,sim_intercept)
        i = self.constant_background_inv(gluc_cur,background)
        return i
    
    # Model6: current = (ref - compartment) * (1/mag * sim_slope +sim_intercept)
    def sim_slope_intercept_constant_compartment_gluc(self,sim,i,params):
        sim_slope = params[0]
        sim_intercept = params[1]
        compartment = params[2]
        dg = self.sim_slope_intercept(sim,i,sim_slope,sim_intercept)
        g = self.constant_compartment(dg,compartment)
        return g
    
    def sim_slope_intercept_constant_compartment_cur(self,sim,g,params):
        sim_slope = params[0]
        sim_intercept = params[1]
        compartment = params[2]
        dg = self.constant_compartment_inv(g,compartment)
        i = self.sim_slope_intercept_inv(sim,dg,sim_slope,sim_intercept)
        return i
    
    # Model7: current = ref * (1/mag * sim_slope +sim_intercept) + (1/mag * background_slope + background_intercept)
    def sim_slope_intercept_linear_background_gluc(self,sim,i,params):
        sim_slope = params[0]
        sim_intercept = params[1]
        background_slope = params[2]
        background_intercept = params[3]
        gluc_cur = self.linear_background(sim,i,background_slope,background_intercept)
        g = self.sim_slope_intercept(sim,gluc_cur,sim_slope,sim_intercept)
        return g
    
    def sim_slope_intercept_linear_background_cur(self,sim,g,params):
        sim_slope = params[0]
        sim_intercept = params[1]
        background_slope = params[2]
        background_intercept = params[3]
        gluc_cur = self.sim_slope_intercept_inv(sim,g,sim_slope,sim_intercept)
        i = self.linear_background_inv(sim,gluc_cur,background_slope,background_intercept)
        return i
    
    # Model8: current = ref * (1/mag * sim_slope +sim_intercept)e + (1/(1/a0+mag/a1))
    def sim_slope_intercept_nonlinear_background_gluc(self,sim,i,params):
        sim_slope = params[0]
        sim_intercept = params[1]
        background_a0 = params[2]
        background_a1 = params[3]
        gluc_cur = self.nonlinear_background(sim,i,background_a0,background_a1)
        g = self.sim_slope_intercept(sim,gluc_cur,sim_slope,sim_intercept)
        return g
    
    def sim_slope_intercept_nonlinear_background_cur(self,sim,g,params):
        sim_slope = params[0]
        sim_intercept = params[1]
        background_a0 = params[2]
        background_a1 = params[3]
        gluc_cur = self.sim_slope_intercept_inv(sim,g,sim_slope,sim_intercept)
        i = self.nonlinear_background_inv(sim,gluc_cur,background_a0,background_a1)
        return i
    
    ################## model based on fmm sensitivity model ################

    # Model9: fmm + constant background
    def fmm_constant_background_gluc(self,sim,i,params):
        sim_slope = params[0]
        vmax = params[1]
        km = params[2]
        background = params[3]
        gluc_cur = self.constant_background(i,background)
        g = self.fmm(sim,gluc_cur,sim_slope,vmax,km)
        return g

    def fmm_constant_background_cur(self,sim,g,params):
        sim_slope = params[0]
        vmax = params[1]
        km = params[2]
        background = params[3]
        gluc_cur = self.fmm_inv(sim,g,sim_slope,vmax,km)
        i = self.constant_background_inv(gluc_cur,background)
        return i
    
    # Model10: fmm + constant compartment
    def fmm_constant_compartment_gluc(self,sim,i,params):
        sim_slope = params[0]
        vmax = params[1]
        km = params[2]
        compartment = params[3]
        dg = self.fmm(sim,i,sim_slope,vmax,km)
        g = self.constant_compartment(dg,compartment)
        return g
    
    def fmm_constant_compartment_cur(self,sim,g,params):
        sim_slope = params[0]
        vmax = params[1]
        km = params[2]
        compartment = params[3]
        dg = self.constant_compartment_inv(g,compartment)
        i = self.fmm_inv(sim,dg,sim_slope,vmax,km)
        return i
    
    # Model11: fmm + linear background
    def fmm_linear_background_gluc(self,sim,i,params):
        sim_slope = params[0]
        vmax = params[1]
        km = params[2]
        background_slope = params[3]
        background_intercept = params[4]
        gluc_cur = self.linear_background(sim,i,background_slope,background_intercept)
        g = self.fmm(sim,gluc_cur,sim_slope,vmax,km)
        return g
    
    def fmm_linear_background_cur(self,sim,g,params):
        sim_slope = params[0]
        vmax = params[1]
        km = params[2]
        background_slope = params[3]
        background_intercept = params[4]
        gluc_cur = self.fmm_inv(sim,g,sim_slope,vmax,km)
        i = self.linear_background_inv(sim,gluc_cur,background_slope,background_intercept)
        return i
    
    # Model12: fmm + nonlinear background
    def fmm_nonlinear_background_gluc(self,sim,i,params):
        sim_slope = params[0]
        vmax = params[1]
        km = params[2]
        background_a0 = params[3]
        background_a1 = params[4]
        gluc_cur = self.nonlinear_background(sim,i,background_a0,background_a1)
        g = self.fmm(sim,gluc_cur,sim_slope,vmax,km)
        return g
    
    def fmm_nonlinear_background_cur(self,sim,g,params):
        sim_slope = params[0]
        vmax = params[1]
        km = params[2]
        background_a0 = params[3]
        background_a1 = params[4]
        gluc_cur = self.fmm_inv(sim,g,sim_slope,vmax,km)
        i = self.nonlinear_background_inv(sim,gluc_cur,background_a0,background_a1)
        return i