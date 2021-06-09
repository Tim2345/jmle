from data_simulation.data_simulation import DataSimulator
from data_simulation.runfile_construction import RunfileConstructor
from jmle.jmle_base import JMLE_base
import numpy as np

n_items = 100
n_persons = 1000

sim = DataSimulator()

items = sim.sample_items_and_persons('normal', 0, 1, n_items)
persons = sim.sample_items_and_persons('normal', .5, 1.25, n_persons)

resp_mat, probs = sim.rasch_response_matrix(persons, items, probabilities=True)


person_status = np.ones(n_persons)
anchor_person_abilities = person_status

anchor_idx = range(20)
item_status = sim.anchor_status(n_items, idx=anchor_idx)
anchor_item_diffs = item_status.copy()
anchor_item_diffs[anchor_idx] = items[anchor_idx]

global_person_status = 1
global_item_status = 0

logit_conv = 0.01
resid_conv = 0.01

jmle = JMLE_base(

    resp_mat=resp_mat,
    person_status=person_status,
    item_status=item_status,
    global_person_status=global_person_status,
    global_item_status=global_item_status,
    anchor_persons=anchor_person_abilities,
    anchor_items=anchor_item_diffs,
    logit_conv=logit_conv,
    resid_conv=resid_conv,
    n_iter=100

)

jmle.estimate()



runfile = RunfileConstructor(
    runfile_name='practice2',
    save_path='Practice2.txt',
    response_matrix=resp_mat,
    item_status=item_status,
    anchor_item_diffs=anchor_item_diffs,
    person_status=person_status,
    anchor_person_abilities=anchor_person_abilities,
    rconv=resid_conv,
    lconv=logit_conv
)

runfile.construct_runfile()



import numpy as np


this = np.arange(10)
that = np.array([0,0,0,0,0,1,1,1,1,1])
these = np.arange(10)-0.5

idx = that==0
np.where(that==0, these, this)

this[idx] = these[idx]
this