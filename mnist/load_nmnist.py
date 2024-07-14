from tonic.datasets.nmnist import NMNIST

root_dir = "./data"
sample_data, label = NMNIST(save_to=root_dir, train=False)[0]

print(f"type of data is: {type(sample_data)}")
print(f"time length of sample data is: {sample_data['t'][-1] - sample_data['t'][0]} micro seconds")
print(f"there are {len(sample_data)} events in the sample data")
print(f"the label of the sample data is: {label}")