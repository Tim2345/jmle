import numpy as np
import re


class RunfileConstructor:

    def __init__(self, runfile_name, save_path, response_matrix, item_status,
                      anchor_item_diffs, person_status, anchor_person_abilities, rconv, lconv):

        self.runfile_name = runfile_name
        self.save_path = save_path
        self.response_matrix = response_matrix
        self.item_status = item_status
        self.anchor_item_diffs = anchor_item_diffs
        self.person_status = person_status
        self.anchor_person_abilities = anchor_person_abilities
        self.rconv = rconv
        self.lconv = lconv
        self.template = """
TITLE = <VALUE>
PERSON = Learners
ITEM = Item
ITEM1 = <VALUE>
NI = <VALUE>
NAME1 = 1
NAMELEN = <VALUE>
CODES = 01
MISSCORE = -1
UDECIM = 2
OUTFIT = N
LINELENGTH = 0
PTBISERIAL = X
PVALUE = YES
CONVERGE = BOTH
RCONV = <VALUE>
LCONV = <VALUE>
MHSLICE = 3
PSUBTOTAL = $S1W2
TABLES = --11---1-1---1-------------1-1
IAFILE=*
<IANCHORFILE>
*
PAFILE=*
<PANCHORFILE>
*
&END
<ITEMIDS>
END LABELS
<RESPONSEDATA>
"""

    def get_anchor_file(self, anchor_status, anchor_diffs):
        anchor_idx = anchor_status==0
        anchor_values = anchor_diffs[anchor_idx]
        anchor_indices = anchor_idx.nonzero()[0]+1

        anchor_string = []
        for idx, val in zip(anchor_indices.astype(str), anchor_values.astype(str)):
            anchor_string.append('\t'.join([idx, val]))

        anchor_string = '\n'.join(anchor_string)

        return anchor_string

    def generate_ids(self, n):
        ids = list(range(n))
        n_len = len(str(n))
        filled_ids = [str(item).zfill(n_len) for item in ids]

        return filled_ids


    def insert_variable(self, variable, insert_val, string, needs_idx_adj=False):
        if needs_idx_adj:
            insert_val = insert_val + 1

        insert_val = str(insert_val)
        reg_find = variable + r'\s*=\s*<VALUE>\s*\n'
        replace_reg = variable + ' = ' + insert_val + '\n'
        replaced_string = re.sub(reg_find, replace_reg, string)

        return replaced_string


    def construct_runfile(self):

        item_ids = self.generate_ids(len(self.item_status))
        item_id_string = '\n'.join(item_ids)
        person_ids = self.generate_ids(len(self.person_status))


        aifile = self.get_anchor_file(self.item_status, self.anchor_item_diffs)
        apfile = self.get_anchor_file(self.person_status, self.anchor_person_abilities)

        # combine person ids with response mat
        responses_string = [''.join(row) for row in self.response_matrix.astype(str)]
        responses_string = [' '.join([p_id, response]) for p_id, response in zip(person_ids, responses_string)]
        responses_string = '\n'.join(responses_string)

        #get index of first response (ITEM1)
        first_item = re.search(r'\s[01N]', responses_string).end()
        # get length of tt_id (NAMELEN)
        id_len = max([len(id) for id in person_ids])

        rf_string = self.insert_variable('TITLE', self.runfile_name, self.template)
        rf_string = self.insert_variable('ITEM1', first_item, rf_string)
        rf_string = self.insert_variable('NI', len(self.item_status), rf_string)
        rf_string = self.insert_variable('NAMELEN', id_len, rf_string)
        rf_string = self.insert_variable('RCONV', self.rconv, rf_string)
        rf_string = self.insert_variable('LCONV', self.lconv, rf_string)
        rf_string = rf_string.replace('<IANCHORFILE>', aifile)
        rf_string = rf_string.replace('<PANCHORFILE>', apfile)
        rf_string = rf_string.replace('<ITEMIDS>', item_id_string)
        rf_string = rf_string.replace('<RESPONSEDATA>', responses_string)

        rf_new = open(self.save_path, 'w')
        rf_new.write(rf_string)
        rf_new.close()


