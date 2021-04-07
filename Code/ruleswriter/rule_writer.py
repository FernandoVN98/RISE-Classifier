class rule_writer:
    def __init__(self, file, rules):
        self.file_to_write = file
        self.rules = rules

    def write(self):
        with open(self.file_to_write, 'w') as output:
            for i in range(len(self.rules)):
                string_to_print = ""
                for x, y in zip(self.rules[i]['conditions'], self.rules[-1]['attributes']):
                    if len(x) == 1:
                        string_to_print = string_to_print + y + " TRUE AND "
                    elif len(x) == 2:
                        string_to_print = string_to_print + y +" = " + str(x[1]) + " AND "
                    elif len(x) == 3:
                        string_to_print = string_to_print + str(x[1]) + " <= " + y + " <= " + str(x[2]) + " AND "

                print("RULE "+str(i)+ " AND", string_to_print+ "CLASS: "+str(self.rules[i]['CR']) + " COVERAGE: "+str(self.rules[i]['coverage'])+" ACC: "+ str(self.rules[i]['accuracy'])+" EndRule", file=output)
