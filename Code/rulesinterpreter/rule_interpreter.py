from numbers import Number


class rule_interpreter:
    def __init__(self, file, q=1, s=1, svdm=None):
        self.file = file
        self.rules = []
        self.accuracies_of_rules = []
        self.q = q
        self.s = s
        self.max = []
        self.min = []
        self.svdm = svdm

    def extract_rules(self):
        with open(self.file) as f:
            lines = f.readlines()
            f.close()
        for line in lines:
            my_rule = {}
            x = line.split(" AND ")
            for word in x:
                count = word.count("<=")
                if 'TRUE' in word:
                    condition = word.split(" TRUE")
                    my_rule[str(condition[0])] = ['TRUE']
                elif count == 2:
                    condition = word.split(" <= ")
                    my_rule[str(condition[1])] = [float(condition[0]), float(condition[2])]
                elif count == 1:
                    condition = word.split(" = ")
                    my_rule[str(condition[0])] = [condition[1]]
                elif '=' in word:
                    condition = word.split(" = ")
                    my_rule[str(condition[0])] = [condition[1]]
                else:
                    condition = word.split(" ")
                    if len(condition) == 7:
                        my_rule[str(condition[0])] = condition[1]
                        my_rule[str(condition[2])] = int(condition[3])
                        my_rule[str(condition[4])] = float(condition[5])
            self.rules.append(my_rule)
        return self.rules

    def compute_distance(self, rule, example):
        actual_dist = 0
        dist = 0
        for i in range(len(example)):
            dist = 0
            if len(rule[list(rule.keys())[i]]) == 0:
                dist = dist + 0
            elif len(rule[list(rule.keys())[i]]) == 2:
                if example[i] > rule[list(rule.keys())[i]][1]:
                    dist = dist + ((example[i] - rule[list(rule.keys())[i]][1]) / (self.max[i] - self.min[i]))
                elif example[i] < rule[list(rule.keys())[i]][0]:
                    dist = dist + ((rule[list(rule.keys())[i]][0] - example[i]) / (self.max[i] - self.min[i]))
            elif len(rule[list(rule.keys())[i]][0]) == 1:
                if example[i] != rule[list(rule.keys())[i]][0] and rule[list(rule.keys())[i]][0] != 'True':
                    for j in self.svdm:
                        if str(example[i]) in j and str(rule[list(rule.keys())[i]][0]) in j:
                            dist = dist + pow(
                                (j[str(i)][str(example[i])] - j[str(i)][str(rule[list(rule.keys())[i]][0])]),
                                self.q)
            actual_dist = actual_dist + pow(dist, self.s)
        return actual_dist

    def compute_nearest_rules(self, examples, y, i):
        distance = 100000000
        for j in range(len(self.rules)):
            actual_distance = self.compute_distance(self.rules[j], examples[i])
            if actual_distance < distance:
                to_calc_lapl = j
                distance = actual_distance
                my_rule = self.rules[j]
            elif actual_distance == distance:
                first_acc = self._get_laplace_correction(self.rules[to_calc_lapl], examples, y)
                second_acc = self._get_laplace_correction(self.rules[j], examples, y)
                if second_acc > first_acc:
                    to_calc_lapl = j
                    my_rule = self.rules[j]
                elif second_acc == first_acc:
                    first_repr = [item for item in self.svdm if item['class'] == self.rules[to_calc_lapl]['CLASS:']][0][
                        'prob']
                    secc_repr = [item for item in self.svdm if item['class'] == self.rules[j]['CLASS:']][0]['prob']
                    if secc_repr > first_repr:
                        to_calc_lapl = j
                        my_rule = self.rules[j]
        return my_rule

    def evaluate_rules(self, es, y, max, min):
        y.reset_index(inplace=True, drop=True)
        '''for i in self.rules[0].keys():
            if i not in ['CLASS:', 'COVERAGE:', 'ACC:']:
                if len(self.rules[0][i]) > 1:
                    seq = [x[i] for x in self.rules]
                    self.min.append(min(x[0] for x in seq))
                    self.max.append(max(x[1] for x in seq))
                else:
                    self.min.append(0)
                    self.max.append(0)'''
        self.max = max
        self.min = min
        correct_classified_items = 0
        for i in range(len(es)):
            rule = self.compute_nearest_rules(es, y, i)
            if isinstance(rule['CLASS:'], Number):
                if y[i] == int(rule['CLASS:']):
                    correct_classified_items = correct_classified_items + 1
            else:
                if y[i] == rule['CLASS:']:
                    correct_classified_items = correct_classified_items + 1

        return correct_classified_items / len(es)

    @staticmethod
    def _get_laplace_correction(rule, examples, y):
        cond = 0
        pos_examples = 0
        neg_examples = 0
        for i in range(len(examples)):
            for k, j in zip(rule.items(), range(len(examples[i]))):
                if len(rule[k[0]]) == 1:
                    if rule[k[0]][0] == examples[i][j]:
                        cond = cond + 1
                elif len(rule[k[0]]) == 2:
                    if rule[k[0]][0] <= examples[i][j] and examples[i][j] <= rule[k[0]][1] or \
                            rule[k[0]][0] == -1:
                        cond = cond + 1
                elif rule[k[0]][0] == 'True':
                    cond = cond + 1
                elif examples[i][j] == '?':
                    cond = cond + 1
            if cond == len(examples[i]):
                if rule['CLASS:'] == y[i]:
                    pos_examples = pos_examples + 1
                else:
                    neg_examples = neg_examples + 1
            cond = 0

        return ((pos_examples + 1) / (pos_examples + neg_examples + len(set(y))))
