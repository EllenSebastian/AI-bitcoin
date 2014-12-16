import json, re

# General code for representing a weighted CSP (Constraint Satisfaction Problem).
# All variables are being referenced by their index instead of their original
# names.
class CSP:
    def __init__(self):
        # Total number of variables in the CSP.
        self.numVars = 0

        # The list of variable names in the same order as they are added. A
        # variable name can be any hashable objects, for example: int, str,
        # or any tuple with hashtable objects.
        self.varNames = []

        # Each entry is the list of domain values that its corresponding
        # variable can take on.
        # E.g. if B \in ['a', 'b'] is the second variable
        # then valNames[1] == ['a', 'b']
        self.valNames = []

        # Each entry is a unary potential table for the corresponding variable.
        # The potential table corresponds to the weight distribution of a variable
        # for all added unary potential functions. The table itself is a list
        # that has the same length as the variable's domain. If there's no
        # unary function, this table is stored as a None object.
        # E.g. if B \in ['a', 'b'] is the second variable, and we added two
        # unary potential functions f1, f2 for B,
        # then unaryPotentials[1][0] == f1('a') * f2('a')
        self.unaryPotentials = []

        # Each entry is a dictionary keyed by the index of the other variable
        # involved. The value is a binary potential table, where each table
        # stores the potential value for all possible combinations of
        # the domains of the two variables for all added binary potneital
        # functions. The table is represented as a 2D list, with size
        # dom(var) x dom(var2).
        #
        # As an example, if we only have two variables
        # A \in ['b', 'c'],  B \in ['a', 'b']
        # and we've added two binary functions f1(A,B) and f2(A,B) to the CSP,
        # then binaryPotentials[0][1][0][0] == f1('b','a') * f2('b','a').
        # binaryPotentials[0][0] should return a key error since a variable
        # shouldn't have a binary potential table with itself.
        #
        # One important thing to note here is that the indices in the potential
        # tables are indexed with respect to its variable's domain. Hence, 'b'
        # will have an index of 0 in A, but an index of 1 in B. Conversely, the
        # first value for A and B may not necessarily represent the same thing.
        # Beaware of the difference when implementing your CSP solver.
        self.binaryPotentials = []

    def add_variable(self, varName, domain):
        """
        Add a new variable to the CSP.
        """
        if varName in self.varNames:
            raise Exception("Variable name already exists: %s" % str(varName))
        var = len(self.varNames)
        self.numVars += 1
        self.varNames.append(varName)
        self.valNames.append(domain)
        self.unaryPotentials.append(None)
        self.binaryPotentials.append(dict())

    def get_neighbor_vars(self, var):
        """
        Returns a list of indices of variables which are neighbors of
        the variable of indec |var|.
        """
        return self.binaryPotentials[var].keys()

    def add_unary_potential(self, varName, potentialFunc):
        """
        Add a unary potential function for a variable. Its potential
        value across the domain will be *merged* with any previously added
        unary potential functions through elementwise multiplication.

        How to get unary potential value given a variable index |var| and
        value index |val|?
        => csp.unaryPotentials[var][val]
        """
        try:
            var = self.varNames.index(varName)
        except ValueError:
            if isinstance(varName, int):
                print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                print '!! Tip:                                                                       !!'
                print '!! It seems you trying to add a unary potential with variable index...        !!'
                print '!! When adding a potential, you should use variable names.                    !!'
                print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            raise

        potential = [float(potentialFunc(val)) for val in self.valNames[var]]
        if self.unaryPotentials[var] is not None:
            assert len(self.unaryPotentials[var]) == len(potential)
            self.unaryPotentials[var] = [self.unaryPotentials[var][i] * \
                potential[i] for i in range(len(potential))]
        else:
            self.unaryPotentials[var] = potential

    def add_binary_potential(self, varName1, varName2, potential_func):
        """
        Takes two variable names and a binary potential function
        |potentialFunc|, add to binaryPotentials. If the two variables already
        had binaryPotentials added earlier, they will be *merged* through element
        wise multiplication.

        How to get binary potential value given a variable index |var1| with value
        index |val1| and variable index |var2| with value index |val2|?
        => csp.binaryPotentials[var1][var2][val1][val2]
        """
        try:
            var1 = self.varNames.index(varName1)
            var2 = self.varNames.index(varName2)
        except ValueError:
            if isinstance(varName1, int):
                print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                print '!! Tip:                                                                       !!'
                print '!! It seems you trying to add a binary potential with variable indices...     !!'
                print '!! When adding a potential, you should use variable names.                    !!'
                print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            raise

        self.update_binary_potential_table(var1, var2,
            [[float(potential_func(val1, val2)) \
                for val2 in self.valNames[var2]] for val1 in self.valNames[var1]])
        self.update_binary_potential_table(var2, var1, \
            [[float(potential_func(val1, val2)) \
                for val1 in self.valNames[var1]] for val2 in self.valNames[var2]])

    def update_binary_potential_table(self, var1, var2, table):
        """
        Private method you can skip for 0c, might be useful for 1c though.
        Update the binary potential table for binaryPotentials[var1][var2].
        If it exists, element-wise multiplications will be performed to merge
        them together.
        """
        if var2 not in self.binaryPotentials[var1]:
            self.binaryPotentials[var1][var2] = table
        else:
            currentTable = self.binaryPotentials[var1][var2]
            assert len(table) == len(currentTable)
            assert len(table[0]) == len(currentTable[0])
            for i in range(len(table)):
                for j in range(len(table[i])):
                    currentTable[i][j] *= table[i][j]

