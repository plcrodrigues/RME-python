
# generic import
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# mne import
from mne import Epochs, pick_types
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne.event import find_events

# pyriemann import
from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import Covariances

# sklearn imports
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# dimreduction imports
from dimred import RDR

def load_data(subject=1):

	# parameters for loading the EEG BCI Physionet dataset
	tmin, tmax = 1., 2.  # time definition of an epoch
	fmin, fmax = 7., 35. # band frequency to filter

	# which task to consider and what each class means
	event_id = dict(hands=2, feet=3)
	runs = [6, 10, 14]  # motor imagery: hands vs feet

	# load the data (download if needed)
	raw_files = [read_raw_edf(f, preload=True, verbose=False) for f in eegbci.load_data(subject, runs, verbose=False)]	             
	raw = concatenate_raws(raw_files)

	picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
	                   
	# apply band-pass filter
	raw.filter(fmin, fmax, method='iir', picks=picks, verbose=False)

	# find the events from the stimulus channel
	events = find_events(raw, shortest_event=0, stim_channel='STI 014', verbose=False)

	# cut the signal into epochs
	epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
	                baseline=None, preload=True, add_eeg_ref=False, verbose=False)
	labels = epochs.events[:, -1] - 2
	
	# get epochs
	epochs_data = 1e6*epochs.get_data()

	# compute covariance matrices
	covs = Covariances(estimator='lwf').fit_transform(epochs_data)

	return covs, labels

def cross_validation(clf, dataset, labels, nfolds=10, test_size=0.2):

    X = dataset
    y = labels

    score = 0
    for _ in tqdm(range(nfolds)): 
        
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size)
        clf.fit(Xtrain, ytrain)
        score = score + clf.score(Xtest, ytest)

    score = score/nfolds

    return score

# to what dimension reduce the covariance matrices
P = 16

# load the data from Physionet Motor Imagery dataset
covs, y = load_data(subject=1)	

pipelines = {}

# create classification pipeline without dimension reduction
mdm   = MDM(metric='riemann')
pipe  = make_pipeline(mdm)
pipelines['MDM'] = pipe

# create classification pipeline with dimension reduction using RME
mdm   = MDM(metric='riemann')
rdr   = RDR(method='rme-uns', n_components=P)
pipe  = make_pipeline(rdr, mdm)
pipelines['RME + MDM'] = pipe

# create classification pipeline with dimension reduction using RME-random
mdm   = MDM(metric='riemann')
rdr   = RDR(method='rme-uns-bm', n_components=P, params={'nmeans': 10, 'npoints': 5})
pipe  = make_pipeline(rdr, mdm)
pipelines['bmRME + MDM'] = pipe

scores = {}
for pipename in pipelines:
	print pipename
	scores[pipename] = cross_validation(pipelines[pipename], covs, y)
print scores




