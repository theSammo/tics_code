import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy.stats import norm
import cma

df = pd.read_csv('https://raw.githubusercontent.com/anne-urai/tics-2020-tutorial/master/data/KS014_train.csv')  # Load .csv file into a pandas DataFrame

df['signed_contrast'] = df['contrast']*df['position']   # We define a new column for "signed contrasts"
df.drop(columns='stim_probability_left', inplace=True)  # Stimulus probability has no meaning for training sessions

print('Total # of trials: ' + str(len(df['trial_num'])))
print('Sessions: ' + str(np.unique(df['session_num'])))
df.head()

def scatterplot_psychometric_data(df,session_num=None,ax=None):
    """Plot psychometric data (optionally, of a chosen training session) as a scatter plot."""
    
    if session_num == None:
        trial_mask = np.ones(len(df['session_num']), dtype=bool) # Select all trials
    else:
        trial_mask = df['session_num'] == session_num # Indexes of trials of the chosen session
    Ntrials = np.sum(trial_mask) # Number of chosen trials
    
    # Count "left" and "right" responses for each signed contrast level
    left_resp = df[(df['response_choice'] == -1) & trial_mask].groupby(['signed_contrast']).count()['trial_num']
    right_resp = df[(df['response_choice'] == 1) & trial_mask].groupby(['signed_contrast']).count()['trial_num']    
    
    if ax == None:
        ax=fig.add_axes([0,0,1,1])
    ax.scatter(left_resp.index,np.zeros(len(left_resp.index)), s=left_resp*10);
    ax.scatter(right_resp.index,np.ones(len(right_resp.index)), s=right_resp*10);
    ax.set_xlabel('Signed contrast (%)')
    ax.set_ylabel('Rightward response')
    if session_num == None:
        ax.set_title('Psychometric data (# trials = ' + str(Ntrials) + ')')        
    else:
        ax.set_title('Psychometric data (session ' + str(session_num) + ', # trials = ' + str(Ntrials) + ')')        
    return ax

def plot_psychometric_data(df,session_num=None,ax=None):
    """Plot psychometric data (optionally, of a chosen training session) as a scatter plot."""
    
    if session_num == None:
        trial_mask = np.ones(len(df['session_num']), dtype=bool) # Select all trials
    else:
        trial_mask = df['session_num'] == session_num # Indexes of trials of the chosen session
    Ntrials = np.sum(trial_mask) # Number of chosen trials
        
    # Count "left" and "right" responses for each signed contrast level
    left_resp = df[(df['response_choice'] == -1) & trial_mask].groupby(['signed_contrast']).count()['trial_num']
    right_resp = df[(df['response_choice'] == 1) & trial_mask].groupby(['signed_contrast']).count()['trial_num']    
    
    frac_resp = right_resp / (left_resp + right_resp)
    err_bar = np.sqrt(frac_resp*(1-frac_resp)/(left_resp + right_resp)) # Why this formula for error bars?
    
    if ax == None:
        ax=fig.add_axes([0,0,1,1])
    ax.errorbar(x=left_resp.index,y=frac_resp,yerr=err_bar,label='data');
    ax.set_xlabel('Signed contrast (%)')
    ax.set_ylabel('Rightward response')
    if session_num == None:
        ax.set_title('Psychometric data (# trials = ' + str(Ntrials) + ')')        
    else:
        ax.set_title('Psychometric data (session ' + str(session_num) + ', # trials = ' + str(Ntrials) + ')')        
    plt.xlim((-105,105))
    plt.ylim((0,1))
    return ax

def psychofun(theta,stim):
    """Psychometric function based on normal CDF and lapses"""
    mu = theta[0]          # bias
    sigma = theta[1]       # slope/noise
    lapse = theta[2]       # lapse rate
    if len(theta) == 4:    # lapse bias
        lapse_bias = theta[3]
    else:
        lapse_bias = 0.5   # if theta has only three elements, assume symmetric lapses
    
    p_right = norm.cdf(stim,loc=mu,scale=sigma)    # Probability of responding "rightwards", without lapses
    p_right = lapse*lapse_bias + (1-lapse)*p_right # Adding lapses

    return p_right

def psychofun_plot(theta,ax):
    """Plot psychometric function"""    
    stim = np.linspace(-100,100,201)   # Create stimulus grid for plotting    
    p_right = psychofun(theta,stim)    # Compute psychometric function values
    ax.plot(stim,p_right,label='model')
    ax.legend()
    return


def psychofun_loglike(theta,df):
    """Log-likelihood for psychometric function model"""
    s_vec = df['signed_contrast'] # Stimulus values
    r_vec = df['response_choice']  # Responses
    
    p_right = psychofun(theta,s_vec)
    
    # Compute summed log likelihood for all rightwards and leftwards responses
    loglike = np.sum(np.log(p_right[r_vec == 1])) + np.sum(np.log(1 - p_right[r_vec == -1]))

    return loglike

def psychofun_repeatlast_loglike(theta,df):
    """Log-likelihood for last-choice dependent psychometric function model"""
    s_vec = np.array(df['signed_contrast']) # Stimulus values
    r_vec = np.array(df['response_choice'])  # Responses
    
    p_last = theta[0] # Probability of responding as last choice
    theta_psy = theta[1:] # Standard psychometric function parameters
        
    p_right = psychofun(theta_psy,s_vec)
    
    # Starting from the 2nd trial, probability of responding equal to the last trial
    p_right[1:] = p_last*(r_vec[0:-1] == 1) + (1-p_last)*p_right[1:] 
    
    # Compute summed log likelihood for all rightwards and leftwards responses
    loglike = np.sum(np.log(p_right[r_vec == 1])) + np.sum(np.log(1 - p_right[r_vec == -1]))

    return loglike

def psychofun_timevarying_loglike(theta,df):
    """Log-likelihood for time-varying psychometric function model"""
    s_vec = np.array(df['signed_contrast']) # Stimulus values
    r_vec = np.array(df['response_choice'])  # Responses
    
    Ntrials = len(s_vec)
    mu_vec = np.linspace(theta[0],theta[4],Ntrials)
    sigma_vec = np.linspace(theta[1],theta[5],Ntrials)
    lapse_vec = np.linspace(theta[2],theta[6],Ntrials)
    lapsebias_vec = np.linspace(theta[3],theta[7],Ntrials)
    
    p_right = np.zeros(Ntrials)
    
    for t in range(0,Ntrials):
        p_right[t] = psychofun([mu_vec[t],sigma_vec[t],lapse_vec[t],lapsebias_vec[t]],s_vec[t])
    
    # Compute summed log likelihood for all rightwards and leftwards responses
    loglike = np.sum(np.log(p_right[r_vec == 1])) + np.sum(np.log(1 - p_right[r_vec == -1]))

    return loglike

#df_blocks = pd.read_csv('https://raw.githubusercontent.com/anne-urai/tics-2020-tutorial/master/data/KS014_biased.csv') 
df_blocks = pd.read_csv('https://raw.githubusercontent.com/anne-urai/tics-2020-tutorial/master/data/KS014_train.csv')
df_blocks['signed_contrast'] = df_blocks.contrast * df_blocks.position

lapseChance = 0.1
lapseBias = 0.5

df = df_blocks

muBottom = -25
muTop = 25
sigBottom = 5
sigTop = 25
results = []

for trial in range(1,16):
    print(trial)
    session_num = trial # Let's use a different session
    bestMu = 5
    bestSigma = 5
    bestLapseChance = .1
    bestLapseBias = 0.5
    theta0 = (bestMu,bestSigma,lapseChance,lapseBias)
    ll = psychofun_loglike(theta0,df[df['session_num'] == session_num])
    for mu in range(muBottom, muTop+5,5):
        if mu == 0:
            continue
        for sigma in range(sigBottom, sigTop+5,5):
            if sigma == 0:
                continue
            for i in range(1,10):
                lapseChance = round(i/10,1)
                for j in range(1,10):
                    lapseBias = round(j/10,1)
                    theta0 = (mu,sigma,lapseChance,lapseBias) 
                    newLL = psychofun_loglike(theta0,df[df['session_num'] == session_num])
                    if newLL > ll:
                        ll = newLL
                        bestMu = mu
                        bestSigma = sigma
                        bestLapseChance = lapseChance
                        bestLapseBias = lapseBias
    print('Log-likelihood value for trial {}: {}'.format(trial, ll))
    print('with mu value {}, sigma value {}, lapse chance {} and lapse bias {}'.format(bestMu, bestSigma, bestLapseChance, bestLapseBias))

    # fig = plt.figure(figsize=(9,4))
    # ax = plot_psychometric_data(df,session_num)
    # theta0 = (bestMu, bestSigma, bestLapseChance, bestLapseBias)
    # psychofun_plot(theta0,ax)

    theta0 = (bestMu, bestSigma, bestLapseChance, bestLapseBias)
    fig = plt.figure(figsize=(9,4))
    ax = plot_psychometric_data(df,session_num)
    psychofun_plot(theta0,ax)
    plt.tight_layout()
    plt.show()

    fig.savefig("trial {} brute force.png".format(trial), dpi=fig.dpi)
    results.append((trial,theta0))
results