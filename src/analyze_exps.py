import numpy as np
import matplotlib.pyplot as plt

def get_content(filename):
	f = open(filename, "r")
	content = f.readlines()
	try:
		i = content.index("Optimizer loaded.\n")
		return list(filter(lambda l: "|" in l, content[i + 2:]))
	except:
		return list(filter(lambda l: "|" in l, content))


def get_x_y(content, skip=True):
	# data = [list(map(lambda l: float(l), line.split("|")[4].split(" ")[2:4])) for line in content]
	# data = np.array(data)
	# mean = data[:,0]
	# std = data[:,1]
	if skip:
		x = [int(line.split(" | ")[1].split(" ")[1]) for line in content][::5]
		y = [float(line.split(" | ")[6].split(" ")[1]) for line in content][::5]
	else:
		x = [int(line.split(" | ")[1].split(" ")[1]) for line in content]
		y = [float(line.split(" | ")[6].split(" ")[1]) for line in content]
	return x, y

def get_results(prefix):
	x = None
	y = []
	for i in range(1, 6):
		content = get_content(prefix + str(i) + ".txt")
		_x, _y = get_x_y(content)
		y.append(_y)
		x = _x

	y = np.array(y)
	y_std = np.std(y, axis=0)
	y_mean = np.mean(y, axis=0)
	return x, y_mean, y_std


x, y, err = get_results("exps/dfa_comp_seed_")
plt.plot(x, y, label="DFA (comp)")
plt.fill_between(x, y-err, y+err, alpha=0.2)

x, y, err = get_results("exps/ast_seed_")
plt.plot(x, y, label="LTL2Action")
plt.fill_between(x, y-err, y+err, alpha=0.2)



plt.legend()
plt.show()

# print()

# plt.plot(x, [0.4 for _ in range(len(x))])

# content = get_content("exps/exp_comp_ast_baseline.txt")
# x, y = get_x_y(content, skip=False)
# plt.plot(x, y)

# content = get_content("exps/exp_comp_new_dfa.txt")
# x, y = get_x_y(content)
# plt.plot(x, y)




# # plt.errorbar(list(range(len(mean))), mean, yerr=std, fmt='o', capsize=5)

# # plt.title('Line Graph with Standard Deviation')
# # plt.xlabel('X Values')
# # plt.ylabel('Y Values')

# # Show the plot
# plt.show()