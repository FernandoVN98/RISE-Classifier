import numpy as np
import pandas as pd
import copy
from numbers import Number


class rise_algorithm():
    def __init__(self, q=1, s=1):
        self.accuracy = 0
        self.q = q
        self.s = s
        self.near_rules = []

    def extract_rules(self, es, y, series):
        self.series = series
        return self._extract_rules(es, y)

    def validation(self, es_val, y_val):
        total_acc = 0
        y_val.reset_index(inplace=True, drop=True)
        for i in range(len(es_val)):
            rule = self._get_nearest_rule_validation(self.rs, es_val, i, y_val)
            if (rule['CR'] == y_val[i]):
                total_acc = total_acc + 1
        return total_acc / len(y_val)

    def _extract_rules(self, es, y):
        y.reset_index(inplace=True, drop=True)
        self.max_values = np.amax(es, axis=0)
        self.min_values = np.amin(es, axis=0)
        self.compute_svdm(es, y)
        self._construct_rules(es, y)
        self.accuracy = self._compute_accuracy(self.rs, es, y, first_accuracy=True)
        print("Initial Accuracy: " + str(self.accuracy))
        if self.accuracy == 0:
            return 0
        else:
            while True:
                flag = False
                for i in range(len(self.rs)):
                    rs_prima = copy.deepcopy(self.rs)
                    nearest_example, index = self._get_nearest_example(es, rs_prima[i], y, i)
                    if index != -1:
                        self._most_specific_generalization(rs_prima[i], nearest_example, index)
                        actual_accuracy, new_near_rules = self._compute_accuracy(self.rs, es, y, changed_rule=i)
                        if actual_accuracy >= self.accuracy:
                            self.accuracy = actual_accuracy
                            flag = True
                            self.rs = rs_prima
                            self.near_rules = new_near_rules
                            if self.check_r_is_unique(i) != True:
                                break
                if flag != True:
                    self.rs[-1]['attributes'] = self.series
                    for k in range(len(self.rs)):
                        self.rs[k]['accuracy'], self.rs[k]['coverage'] = self._compute_rule_acc(self.rs[k], k, es, y)
                    break
            print("Final accuracy: " + str(self.accuracy))
            return self

    def _construct_rules(self, es, y):
        self.rs = []
        for i in range(len(es)):
            rule = {'CR': y[i]}
            conditions = []
            for j in es[i]:
                if isinstance(j, Number):
                    if j == -1:
                        conditions.append(['number', 2, -1])
                    else:
                        conditions.append(['number', j, j])
                else:
                    conditions.append(['symbolic', j])
                    '''if j != '?':
                        conditions.append(['symbolic', j])
                    else:
                        conditions.append(['True'])'''
            rule['conditions'] = conditions
            rule['matched_element'] = [i]
            self.rs.append(rule)

    def _compute_rule_acc(self, rule, k, es, y):
        acc = 0
        total_matches = 0
        for i in range(len(self.near_rules)):
            if self.near_rules[i]['rule'] == k:
                total_matches = total_matches + 1
                if y[i] == rule['CR']:
                    acc = acc + 1
        if total_matches == 0:
            return 0, 0
        else:
            return (acc / total_matches), total_matches

    def _get_nearest_example(self, examples, rule, y, i):
        my_example = 0
        index = -1
        actualdistance = 10000000000
        for i in range(len(examples)):
            if i not in rule['matched_element'] and rule['CR'] == y[i]:
                distance = self.compute_distance(rule, examples[i])
                if distance < actualdistance:
                    my_example = examples[i]
                    index = i
                    actualdistance = distance
        return my_example, index

    def _get_nearest_rule_validation(self, rs, examples, i, y):
        my_rule = 0
        actualdistance = 10000000000
        for j in range(len(self.rs)):
            distance = self.compute_distance(rs[j], examples[i])
            if distance < actualdistance:
                to_calc_lapl = j
                actualdistance = distance
                my_rule = self.rs[j]
            elif distance == actualdistance:
                first_acc = self._get_laplace_correction_validation(self.rs[to_calc_lapl], examples, y)
                second_acc = self._get_laplace_correction_validation(self.rs[j], examples, y)
                if second_acc > first_acc:
                    to_calc_lapl = j
                    my_rule = self.rs[j]
                elif second_acc == first_acc:
                    first_repr = [item for item in self.svdm if item['class'] == self.rs[to_calc_lapl]['CR']][0][
                        'prob']
                    secc_repr = [item for item in self.svdm if item['class'] == self.rs[j]['CR']][0]['prob']
                    if secc_repr > first_repr:
                        to_calc_lapl = j
                        my_rule = self.rs[j]
        return my_rule

    def _get_nearest_rule(self, rs, examples, i, y):
        my_rule = 0
        actualdistance = 10000000000

        for j in range(len(self.rs)):
            distance = self.compute_distance(rs[j], examples[i])
            if distance < actualdistance:
                if self.rs[j]['matched_element'][0] != i or len(self.rs[j]['matched_element']) > 1:
                    to_calc_lapl = j
                    actualdistance = distance
                    my_rule = self.rs[j]
                    if (len(self.near_rules) <= i):
                        self.near_rules.append({'example': i, 'rule': j, 'distance': actualdistance})
                    else:
                        self.near_rules[i] = {'example': i, 'rule': j, 'distance': actualdistance}
            elif distance == actualdistance:
                if self.rs[j]['matched_element'][0] != i or len(self.rs[j]['matched_element']) > 1:
                    first_acc = self._get_laplace_correction(self.rs[to_calc_lapl], examples, y)
                    second_acc = self._get_laplace_correction(self.rs[j], examples, y)
                    if second_acc > first_acc:
                        to_calc_lapl = j
                        my_rule = self.rs[j]
                        if (len(self.near_rules) <= i):
                            self.near_rules.append({'example': i, 'rule': j, 'distance': actualdistance})
                        else:
                            self.near_rules[i] = {'example': i, 'rule': j, 'distance': actualdistance}
                    elif second_acc == first_acc:
                        first_repr = [item for item in self.svdm if item['class'] == self.rs[to_calc_lapl]['CR']][0][
                            'prob']
                        secc_repr = [item for item in self.svdm if item['class'] == self.rs[j]['CR']][0]['prob']
                        if secc_repr > first_repr:
                            to_calc_lapl = j
                            my_rule = self.rs[j]
                            if (len(self.near_rules) <= i):
                                self.near_rules.append({'example': i, 'rule': j, 'distance': actualdistance})
                            else:
                                self.near_rules[i] = {'example': i, 'rule': j, 'distance': actualdistance}
        return my_rule

    def check_r_is_unique(self, i):
        for j in range(len(self.rs)):
            if i != j:
                if self.equal_dicts(self.rs[j], self.rs[i], ["matched_element"]):
                    self.rs[j]['matched_element'].extend(self.rs[i]['matched_element'])
                    self.rs[j]['matched_element'] = list(set(self.rs[j]['matched_element']))
                    self.update_identifier_near_rule(i, j)
                    self.rs.pop(i)
                    return False
        return True

    @staticmethod
    def equal_dicts(d1, d2, ignore_keys):
        ignored = set(ignore_keys)
        for k1, v1 in d1.items():
            if k1 not in ignored and (k1 not in d2 or d2[k1] != v1):
                return False
        for k2, v2 in d2.items():
            if k2 not in ignored and k2 not in d1:
                return False
        return True

    def update_identifier_near_rule(self, i, j):
        for k in range(len(self.near_rules)):
            if self.near_rules[k]['rule'] > i:
                self.near_rules[k]['rule'] = self.near_rules[k]['rule'] - 1
            elif self.near_rules[k]['rule'] == i:
                self.near_rules[k]['rule'] = j

    def compute_svdm(self, es, y):
        different_classes = list(set(y))
        df = pd.DataFrame(data=es)
        df["result"] = y
        all_classes = []
        for j in different_classes:
            my_class = {
                "class": j
            }
            my_class["occurrence"] = len(df.loc[df["result"] == j])
            i = 0
            subset = df.loc[df["result"] == j]
            for row in subset:
                values_list = []
                my_indexes = subset[row].value_counts()
                for value, index in zip(my_indexes, my_indexes.index):
                    if i < len(es[0]):
                        values_list.append([index, value])
                        my_class[str(i)] = values_list
                i = i + 1
            all_classes.append(my_class)
        prob_list = []
        prob_classes = {}
        prob_x_class = {}
        var_i = {}
        for i in range(len(es)):
            var_i[str(i)] = []
        for a_class in all_classes:
            prob_classes['class'] = a_class['class']
            prob_classes['prob'] = (a_class['occurrence'] / (len(es)))
            for i in range(len(es[0])):
                for value in a_class[str(i)]:
                    if type(es[0][i]) == str:
                        prob_x_class[str(value[0])] = value[1] / (a_class['occurrence']) * a_class['occurrence'] / \
                                                      df[i].value_counts()[str(value[0])]
                    else:
                        prob_x_class[str(value[0])] = 0
                prob_classes[str(i)] = prob_x_class
                prob_x_class = {}
            prob_list.append(prob_classes)
            prob_classes = {}
        self.svdm = prob_list

    def compute_distance(self, rule, example):
        actual_dist=0
        for i in range(len(rule['conditions'])):
            dist = 0
            if rule['conditions'][i][0] == 'number':
                if example[i] != -1 and rule['conditions'][i][1] != 2 and rule['conditions'][i][2] != -1:
                    if example[i] > rule['conditions'][i][2]:
                        dist = dist + (
                                (example[i] - rule['conditions'][i][2]) / (self.max_values[i] - self.min_values[i]))
                    elif example[i] < rule['conditions'][i][1]:
                        dist = dist + (
                                (rule['conditions'][i][1] - example[i]) / (self.max_values[i] - self.min_values[i]))
            elif rule['conditions'][i][0] == 'symbolic':
                if example[i] != rule['conditions'][i][1]:
                    for j in self.svdm:
                        if str(example[i]) in j and str(rule['conditions'][i][1]) in j:
                            dist = dist + pow((j[str(i)][str(example[i])] - j[str(i)][str(rule['conditions'][i][1])]),
                                              self.q)
            actual_dist=actual_dist+pow(dist,self.s)
        return actual_dist

    def _compute_accuracy(self, rs, es, y, first_accuracy=False, changed_rule=None):
        if first_accuracy:
            total_acc = 0
            for i in range(len(es)):
                rule = self._get_nearest_rule(rs, es, i, y)
                if (rule['CR'] == y[i]):
                    total_acc = total_acc + 1
            return total_acc / len(y)
        else:
            near_rules = copy.deepcopy(self.near_rules)
            total_acc = 0
            for i in range(len(es)):
                actual_dist = self.compute_distance(rs[changed_rule], es[i])
                if actual_dist < near_rules[i][
                    'distance']:
                    near_rules[i]['rule'] = changed_rule
                    near_rules[i]['distance'] = actual_dist
                    if rs[changed_rule]['CR'] == y[i]:
                        total_acc = total_acc + 1
                elif actual_dist == near_rules[i][
                    'distance']:
                    first_acc = self._get_laplace_correction(self.rs[near_rules[i]['rule']], es, y)
                    second_acc = self._get_laplace_correction(rs[changed_rule], es, y)
                    if second_acc > first_acc:
                        near_rules[i]['rule'] = changed_rule
                        near_rules[i]['distance'] = actual_dist
                        if rs[changed_rule]['CR'] == y[i]:
                            total_acc = total_acc + 1
                    elif second_acc == first_acc:
                        first_repr = \
                            [item for item in self.svdm if item['class'] == self.rs[near_rules[i]['rule']]['CR']][0][
                                'prob']
                        secc_repr = [item for item in self.svdm if item['class'] == self.rs[changed_rule]['CR']][0][
                            'prob']
                        if secc_repr > first_repr:
                            near_rules[i]['rule'] = changed_rule
                            near_rules[i]['distance'] = actual_dist
                            if rs[changed_rule]['CR'] == y[i]:
                                total_acc = total_acc + 1
                        else:
                            if (rs[near_rules[i]['rule']]['CR'] == y[i]):
                                total_acc = total_acc + 1
                    else:
                        if (rs[near_rules[i]['rule']]['CR'] == y[i]):
                            total_acc = total_acc + 1
                else:
                    if (rs[near_rules[i]['rule']]['CR'] == y[i]):
                        total_acc = total_acc + 1
            return total_acc / len(y), near_rules

    @staticmethod
    def _get_laplace_correction_validation(rule, examples, y):
        cond = 0
        pos_examples = 0
        neg_examples = 0
        for i in range(len(examples)):
            for j in range(len(examples[i])):
                if rule['conditions'][j][0] == 'symbolic':
                    if rule['conditions'][j][1] == examples[i][j]:
                        cond = cond + 1
                elif rule['conditions'][j][0] == 'number':
                    if rule['conditions'][j][1] <= examples[i][j] and examples[i][j] <= rule['conditions'][j][2] or \
                            rule['conditions'][j][1] == -1:
                        cond = cond + 1
                elif rule['conditions'][j][0] == 'True':
                    cond = cond + 1
                elif examples[i][j] == '?':
                    cond = cond + 1
            if cond == len(examples[i]):
                if rule['CR'] == y[i]:
                    pos_examples = pos_examples + 1
                else:
                    neg_examples = neg_examples + 1
            cond = 0

        return ((pos_examples + 1) / (pos_examples + neg_examples + len(set(y))))

    @staticmethod
    def _get_laplace_correction(rule, examples, y):
        cond = 0
        pos_examples = 0
        neg_examples = 0
        for i in range(len(examples)):
            for j in range(len(examples[i])):
                if rule['conditions'][j][0] == 'symbolic':
                    if rule['conditions'][j][1] == examples[i][j]:
                        cond = cond + 1
                elif rule['conditions'][j][0] == 'number':
                    if rule['conditions'][j][1] <= examples[i][j] and examples[i][j] <= rule['conditions'][j][2] or \
                            rule['conditions'][j][1] == -1:
                        cond = cond + 1
                elif rule['conditions'][j][0] == 'True':
                    cond = cond + 1
                elif examples[i][j] == '?':
                    cond = cond + 1
            if cond == len(examples[i]):
                if rule['CR'] == y[i]:
                    pos_examples = pos_examples + 1
                else:
                    neg_examples = neg_examples + 1
            cond = 0

        return ((pos_examples + 1) / (pos_examples + neg_examples + len(set(y))))

    @staticmethod
    def _most_specific_generalization(rule, example, index):
        for i in range(len(example)):
            if len(rule['conditions'][i]) > 0:
                if rule['conditions'][i][0] == 'symbolic':
                    if rule['conditions'][i][1] != example[i] and example[i] != '?':
                        rule['conditions'][i] = ['True']
                elif rule['conditions'][i][0] == 'number' and example[i] != -1:
                    if rule['conditions'][i][1] > example[i]:
                        rule['conditions'][i][1] = example[i]
                    if rule['conditions'][i][2] < example[i]:
                        rule['conditions'][i][2] = example[i]
        rule['matched_element'].append(index)
        return rule
