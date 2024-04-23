# %%
import pickle

with open('../data/NIMBLE/NIMBLE_DICT_9137.pkl', 'rb') as fp:
    nimble_dict = pickle.load(fp)

# print(nimble_dict.keys())
for k, v in nimble_dict.items():
    if isinstance(v, int):
        print(k, v)
    else:
        print(k,v.shape)


# %%
