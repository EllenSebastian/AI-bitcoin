# for MCV, my domains never change - how to find the most constrained variable>
# for AC3, what if my new domain and old domain don't overlap??
import collections, util, copy, pdb, Queue

############################################################
# Problem 0
def unique(a):
    k=0
    while k < len(a):
        if a[k] in a[k+1:]:
            a.pop(k)
        else:
            k=k+1
    return a

def create_chain_csp(n):
    # same domain for each variable
    domain = [0, 1]
    # name variables as x_1, x_2, ..., x_n
    variables = ['x%d'%i for i in range(1, n+1)]
    csp = util.CSP()
    # Problem 0c
    for var in variables: 
        csp.add_variable(var, domain)
    def potential(a,b): 
        return (a + b) == 1 
    
    for i in xrange(1, n): 
        csp.add_binary_potential('x%d'%i, 'x%d'%(i+1), potential)

    return csp



############################################################
# Problem 1

def create_nqueens_csp(n = 8):
    csp = util.CSP()
    # Problem 1a
    queens = ['q%d'%i for i in range(0, n)]
    def potential(p1, p2): 
        if (p1[0] == p2[0]) or (p1[1] == p2[1]) or (abs(p2[0] - p1[0]) == abs(p2[1] - p1[1])): 
            #print 'returning 0 for {0}, {1}'.format(p1, p2)
            return 0
        #print 'returning 1 for {0}, {1}'.format(p1, p2)
        return 1 
    col = 0
    for queen in queens: 
        domain = []
        for j in xrange(n):
            domain.append([col,j])
        csp.add_variable(queen, domain)
        col += 1 
    for q1 in xrange(0,n): 
        for q2 in xrange(0, n): 
            if q1 != q2: 
                #print 'adding {0}, {1}'.format('q%d'%q1, 'q%d'%q2)
                csp.add_binary_potential('q%d'%q1, 'q%d'%q2, potential)
    return csp

# A backtracking algorithm that solves weighted CSP.
# Usage:
#   search = BacktrackingSearch()
#   search.solve(csp)
class BacktrackingSearch():

    def reset_results(self):
        """
        This function resets the statistics of the different aspects of the
        CSP solver. We will be using the values here for grading, so please
        do not make any modification to these variables.
        """
        # Keep track of the best assignment and weight found.
        self.optimalAssignment = {}
        self.optimalWeight = 0

        # Keep track of the number of optimal assignments and assignments. These
        # two values should be identical when the CSP is unweighted or only has binary
        # weights.
        self.numOptimalAssignments = 0
        self.numAssignments = 0

        # Keep track of the number of times backtrack() gets called.
        self.numOperations = 0

        # Keep track of the number of operations to get to the very first successful
        # assignment (doesn't have to be optimal).
        self.firstAssignmentNumOperations = 0

        # List of all solutions found.
        self.allAssignments = []

    def print_stats(self):
        """
        Prints a message summarizing the outcome of the solver.
        """
        if self.optimalAssignment:
            print "Found %d optimal assignments with weight %f in %d operations" % \
                (self.numOptimalAssignments, self.optimalWeight, self.numOperations)
            print "First assignment took %d operations" % self.firstAssignmentNumOperations
        else:
            print "No solution was found."

    def get_delta_weight(self, assignment, var, val):
        """
        Given a CSP, a partial assignment, and a proposed new value for a variable,
        return the change of weights after assigning the variable with the proposed
        value.

        @param assignment: A list of current assignment. len(assignment) should
            equal to self.csp.numVars. Unassigned variables have None values, while an
            assigned variable has the index of the value with respect to its
            domain. e.g. if the domain of the first variable is [5,6], and 6
            was assigned to it, then assignment[0] == 1.
        @param var: Index of an unassigned variable.
        @param val: Index of the proposed value with respect to |var|'s domain.

        @return w: Change in weights as a result of the proposed assignment. This
            will be used as a multiplier on the current weight.
        """
        assert assignment[var] is None
        w = 1.0
        if self.csp.unaryPotentials[var]:
            w *= self.csp.unaryPotentials[var][val]
            if w == 0: return w
        for var2, potential in self.csp.binaryPotentials[var].iteritems():
            if assignment[var2] == None: continue  # Not assigned yet
            w *= potential[val][assignment[var2]]
            if w == 0: return w
        return w

    def solve(self, csp, mcv = False, lcv = False, mac = False):
        """
        Solves the given weighted CSP using heuristics as specified in the
        parameter. Note that unlike a typical unweighted CSP where the search
        terminates when one solution is found, we want this function to find
        all possible assignments. The results are stored in the variables
        described in reset_result().

        @param csp: A weighted CSP.
        @param mcv: When enabled, Most Constrained Variable heuristics is used.
        @param lcv: When enabled, Least Constraining Value heuristics is used.
        @param mac: When enabled, AC-3 will be used after each assignment of an
            variable is made.
        """
        # CSP to be solved.
        self.csp = csp

        # Set the search heuristics requested asked.
        self.mcv = mcv
        self.lcv = lcv
        self.mac = mac

        # Reset solutions from previous search.
        self.reset_results()

        # The list of domains of every variable in the CSP. Note that we only
        # use the indices of the values. That is, if the domain of a variable
        # A is [2,3,5], then here, it will be stored as [0,1,2]. Original domain
        # name/value can be obtained from self.csp.valNames[A]
        self.domains = [list(range(len(domain))) for domain in self.csp.valNames]

        # Perform backtracking search.
        self.backtrack([None] * self.csp.numVars, 0, 1)

        # Print summary of solutions.
        self.print_stats()

    def backtrack(self, assignment, numAssigned, weight):
        """
        Perform the back-tracking algorithms to find all possible solutions to
        the CSP.

        @param assignment: A list of current assignment. len(assignment) should
            equal to self.csp.numVars. Unassigned variables have None values, while an
            assigned variable has the index of the value with respect to its
            domain. e.g. if the domain of the first variable is [5,6], and 6
            was assigned to it, then assignment[0] == 1.
        @param numAssigned: Number of currently assigned variables
        @param weight: The weight of the current partial assignment.
        """

        self.numOperations += 1
        assert weight > 0
        if numAssigned == self.csp.numVars:
            # A satisfiable solution have been found. Update the statistics.
            self.numAssignments += 1
            newAssignment = {}
            for var in range(self.csp.numVars):
                newAssignment[self.csp.varNames[var]] = self.csp.valNames[var][assignment[var]]

            self.allAssignments.append(newAssignment)

            if len(self.optimalAssignment) == 0 or weight >= self.optimalWeight:
                if weight == self.optimalWeight:
                    self.numOptimalAssignments += 1
                else:
                    self.numOptimalAssignments = 1
                self.optimalWeight = weight
                self.optimalAssignment = newAssignment
                if self.firstAssignmentNumOperations == 0:
                    self.firstAssignmentNumOperations = self.numOperations
            return

        # Select the index of the next variable to be assigned.
        var = self.get_unassigned_variable(assignment)

        # Least constrained value (LCV) is not used in this assignment
        # so just use the original ordering
        ordered_values = self.domains[var]

        # Continue the backtracking recursion using |var| and |ordered_values|.
        if not self.mac:
            # When arc consistency check is not enabled.
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(assignment, var, val)
                print(",({0},{1}):{2}".format(self.csp.varNames[var], val, deltaWeight))
                #pdb.set_trace()
                if deltaWeight > 0:
                    assignment[var] = val
                    self.backtrack(assignment, numAssigned + 1, weight * deltaWeight)
                    assignment[var] = None
        else:
            # Arc consistency check is enabled.
            # Problem 1c: skeleton code for AC-3
            # You need to implement arc_consistency_check().
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(assignment, var, val)
                if deltaWeight > 0:
                    assignment[var] = val
                    # create a deep copy of domains as we are going to look
                    # ahead and change domain values
                    localCopy = copy.deepcopy(self.domains)
                    # fix value for the selected variable so that hopefully we
                    # can eliminate values for other variables
                    #pdb.set_trace()
                    self.domains[var] = [val]

                    # enforce arc consistency
                    self.arc_consistency_check(var)

                    self.backtrack(assignment, numAssigned + 1, weight * deltaWeight)
                    # restore the previous domains
                    self.domains = localCopy
                    assignment[var] = None

    def get_unassigned_variable(self, assignment):
        """
        Given a partial assignment, return the index of a currently unassigned
        variable.

        @param assignment: A list of current assignment. This is the same as
            what you've seen so far.

        @return var: Index of a currently unassigned variable.
        """
        if self.mcv:
            min_index = -1
            min_count = float('inf')
            counts = [0] * len(self.domains) # how many are free
            for var in xrange(len(self.domains)):
                for val in self.domains[var]:
                    if assignment[var] is None:
                        if self.get_delta_weight(assignment, var, val) != 0:
                            counts[var] += 1
            for count_i in xrange(len(counts)): 
                if assignment[count_i] is None and counts[count_i] < min_count: 
                    min_index = count_i
                    min_count = counts[count_i]
            #pdb.set_trace()
            if min_index != -1: 
                return min_index
        
        # Select a variable without any heuristics.
        for var in xrange(len(assignment)):
            if assignment[var] is None: return var



    # PROBLEM! This leads to often having EMPTY domains. --> problem?
    def arc_consistency_check(self, var):
        """
        Perform the AC-3 algorithm. The goal is to reduce the size of the
        domain values for the unassigned variables based on arc consistency.

        @param var: The index of variable whose value has just been set.
        """
        # Problem 1c
        # Hint: How to get indices of variables neighboring variable at index |var|?
        # => for var2 in self.csp.get_neighbor_vars(var):
        #       # use var2
        #
        # Hint: How to check if two values are inconsistent?
        # For unary potentials (var1 is the index, val1 is the value):
        #   => self.csp.unaryPotentials[var1][val1] == 0
        #
        # For binary potentials (var1 and var2 are indices, val1 and val2 are values)::
        #   => self.csp.binaryPotentials[var1][var2][val1][val2] == 0
        #   (self.csp.binaryPotentials[var1][var2] returns a nested dict of all assignments)
        q = Queue.Queue()
        # the index of the xi, and its value(s)
        q.put(var)
        n_domains = len(self.domains)
        #print 'given new constraint {0} = {1} = {2}'.format(var, self.domains[var], self.csp.valNames[var][self.domains[var][0]])
        #print 'domains start: ', self.domains
        if self.domains == [[3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]] and var == 0 and self.domains[var] == [3]:
            pdb.set_trace()
        while not q.empty(): 
            xi = q.get()
            xi_domain = list(self.domains[xi])
            for neighbor_j in self.csp.get_neighbor_vars(xi): 
                domain_j = self.domains[neighbor_j]
                potentials = self.csp.binaryPotentials[xi][neighbor_j]
                # here we are looking at self.csp.binaryPotentials[xi][neighbor_j][xi] to find what neighbor_j is allowed to be given that Xi=xi
                changed = False
                #print 'self.domains[{0}] = {1}'.format(neighbor_j, self.domains[neighbor_j])
                domain_for_loop = list(self.domains[neighbor_j])
                for j_choice in domain_for_loop: 
                    #pdb.set_trace()
                    unary_bad = self.csp.unaryPotentials[neighbor_j] is not None and self.csp.unaryPotentials[neighbor_j][j_choice] == 0
                    if unary_bad or sum([potentials[xi_choice][j_choice] for xi_choice in xi_domain]) == 0: 
                        self.domains[neighbor_j].remove(j_choice)
                        #print 'self.domains[{0}].remove({1})'.format(neighbor_j, domain_choice)
                        if not changed: 
                            changed = True
                            q.put(neighbor_j)
                #print 'self.domains[{0}] = {1}'.format(neighbor_j, self.domains[neighbor_j])
                for choice in self.domains[neighbor_j]: 
                    val = self.csp.valNames[neighbor_j][choice]
                    #print val 
        #print 'domains end', self.domains
        #print ''

############################################################
# Problem 2

def get_or_variable(csp, name, variables, value):
    """
    Create a new variable with domain [True, False] that can only be assigned to
    True iff at least one of the |variables| is assigned to |value|. You should
    add any necessary intermediate variables, unary potentials, and binary
    potentials to achieve this. Then, return the name of this variable.

    @param name: Prefix of all the variables that are going to be added.
        Can be any hashable objects. For every variable |var| added in this
        function, it's recommended to use a naming strategy such as
        ('or', |name|, |var|) to avoid conflicts with other variable names.
    @param variables: A list of variables in the CSP that are participating
        in this OR function. Note that if this list is empty, then the returned
        variable created should never be assigned to True.
    @param value: For the returned OR variable being created to be assigned to
        True, at least one of these variables must have this value.

    @return result: The OR variable's name. This variable should have domain
        [True, False] and constraints s.t. it's assigned to True iff at least
        one of the |variables| is assigned to |value|.
    """
    result = ('or', name, 'aggregated')
    csp.add_variable(result, [True, False])

    # no input variable, result should be False
    if len(variables) == 0:
        csp.add_unary_potential(result, lambda val: not val)
        return result

    # Let the input be n variables X0, X1, ..., Xn.
    # After adding auxiliary variables, the factor graph will look like this:
    #
    # ^--A0 --*-- A1 --*-- ... --*-- An --*-- result--^
    #    |        |                  |
    #    *        *                  *
    #    |        |                  |
    #    X0       X1                 Xn
    #
    # where each "--*--" is a binary constraint
    # and each "--^" is a unary constraint.
    for i, X_i in enumerate(variables):
        # create auxiliary variable for variable i
        # use systematic naming to avoid naming collision
        A_i = ('or', name, i)
        # domain values:
        # - [ prev ]: condition satisfied by some previous X_j
        # - [equals]: condition satisfied by X_i
        # - [  no  ]: condition not satisfied yet
        csp.add_variable(A_i, ['prev', 'equals', 'no'])

        # incorporate information from X_i
        def potential(val, b):
            if (val == value): return b == 'equals'
            return b != 'equals'
        csp.add_binary_potential(X_i, A_i, potential)

        if i == 0:
            # the first auxiliary variable, its value should never
            # be 'prev' because there's no X_j before it
            csp.add_unary_potential(A_i, lambda b: b != 'prev')
        else:
            # consistency between A_{i-1} and A_i
            def potential(b1, b2):
                if b1 in ['equals', 'prev']: return b2 != 'no'
                return b2 != 'prev'
            csp.add_binary_potential(('or', name, i - 1), A_i, potential)

    # consistency between A_n and result
    # hacky: reuse A_i because of python's loose scope
    csp.add_binary_potential(A_i, result, lambda val, res: res == (val != 'no'))
    return result

def get_sum_variable(csp, name, variables, maxSum):
    """
    Given a list of |variables| each with non-negative integer domains,
    returns the name of a new variable with domain [0, maxSum], such that
    it's consistent with the value |n| iff the assignments for |variables|
    sums to |n|.

    @param name: Prefix of all the variables that are going to be added.
        Can be any hashable objects. For every variable |var| added in this
        function, it's recommended to use a naming strategy such as
        ('sum', |name|, |var|) to avoid conflicts with other variable names.
    @param variables: A list of variables that are already in the CSP that
        have non-negative integer values as its domain.
    @param maxSum: An integer indicating the maximum sum value allowed.

    @return result: The name of a newly created variable with domain
        [0, maxSum] such that it's consistent with an assignment of |n|
        iff the assignment of |variables| sums to |n|.
    """
    # Problem 2b
    # reutrn a variable that's consistent only when it's the second index of a3
    # need to generalize this 

    def start_with_0(A1_val): 
        return int(A1_val[0] == 0)

    def X_A_consistency(X_val, A_val): 
        return int(A_val[1] == X_val + A_val[0])

    def Ai_Aj_consistency(Ai_val, Aj_val): 
        return int(Aj_val[0] == Ai_val[1])

    def sum_matches_last(A_last_val, sum_val): 
        return sum_val == A_last_val[1]

    domain = []
    for i in range(maxSum + 1):
        for j in range(i,maxSum + 1): 
            domain.append([i,j])

    for var_i in xrange(len(variables)):
        var = variables[var_i]
        new_var = (var, 'counter') 
        csp.add_variable(new_var, domain)
        csp.add_binary_potential(var, new_var, X_A_consistency)
        if var_i != 0: 
            prev_var = (variables[var_i - 1], 'counter')
            csp.add_binary_potential(prev_var, new_var, Ai_Aj_consistency)
            #print 'csp.add_binary_potential({0}, {1}, Ai_Aj_consistency)'.format(prev_var, new_var)
        else: 
            #print 'csp.add_binary_potential({0}, start_with_0)'.format(new_var)
            csp.add_unary_potential(new_var, start_with_0)
        if var_i == len(variables) - 1: 
            csp.add_variable(name, range(0, maxSum + 1))
            #print 'csp.add_binary_potential({0}, {1}, sum_matches_last)'.format(new_var, name)
            csp.add_binary_potential(new_var, name, sum_matches_last)
        #print 'csp.add_binary_potential({0}, {1}, X_A_consistency)'.format(var, new_var)
    return name 

############################################################
# Problem 3

# A class providing methods to generate CSP that can solve the course scheduling
# problem.
class SchedulingCSPConstructor():

    def __init__(self, bulletin, profile):
        """
        Saves the necessary data.

        @param bulletin: Stanford Bulletin that provides a list of courses
        @param profile: A student's profile and requests
        """
        self.bulletin = bulletin
        self.profile = profile

    def add_variables(self, csp):
        """
        Adding the variables into the CSP. Each variable, (req, quarter),
        can take on the value of one of the courses requested in req or None.
        For instance, for quarter='Aut2013', and a request object, req, generated
        from 'CS221 or CS246', then (req, quarter) should have the domain values
        ['CS221', 'CS246', None]. Conceptually, if var is assigned 'CS221'
        then it means we are taking 'CS221' in 'Aut2013'. If it's None, then
        we not taking either of them in 'Aut2013'.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for req in self.profile.requests:
            for quarter in self.profile.quarters:
                csp.add_variable((req, quarter), req.cids + [None])

    def add_bulletin_constraints(self, csp):
        """
        Add the constraints that a course can only be taken if it's offered in
        that quarter.

        @param csp: The CSP where the additional constraints will be added to.
        """
        # foreach request and quarter, allow it if its offered
        for req in self.profile.requests:
            for quarter in self.profile.quarters:
                csp.add_unary_potential((req, quarter), \
                    lambda cid: cid is None or \
                        self.bulletin.courses[cid].is_offered_in(quarter))

    def add_norepeating_constraints(self, csp):
        """
        No course can be repeated. Coupling with our problem's constraint that
        only one of a group of requested course can be taken, this implies that
        every request can only be satisfied in at most one quarter.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for req in self.profile.requests:
            for quarter1 in self.profile.quarters:
                for quarter2 in self.profile.quarters:
                    if quarter1 == quarter2: continue
                    csp.add_binary_potential((req, quarter1), (req, quarter2), \
                        lambda cid1, cid2: cid1 is None or cid2 is None)

    def get_basic_csp(self):
        """
        Return a CSP that only enforces the basic constraints that a course can
        only be taken when it's offered and that a request can only be satisfied
        in at most one quarter.

        @return csp: A CSP where basic variables and constraints are added.
        """
        csp = util.CSP()
        self.add_variables(csp)
        self.add_bulletin_constraints(csp)
        self.add_norepeating_constraints(csp)
        return csp

    def add_quarter_constraints(self, csp):
        """
        If the profile explicitly wants a request to be satisfied in some given
        quarters, e.g. Aut2013, then add constraints to not allow that request to
        be satisfied in any other quarter.

        @param csp: The CSP where the additional constraints will be added to.
        """
        # Problem 3a
        # foreach (request, quarter) that is not allowed, add a unary potential that always returns 0 so it's not allowed.
        quarters = set(self.profile.quarters)
        def quarters_match(arg):
            return 1 
            #print 'arg', arg
            #quarter = arg[1]
            #request = arg[0]
            #return int(quarter in request.quarters)
        def quarters_dont_match(arg): 
            return int(arg is None)

        # for variables where the quarters match, return 1 always
        # for variables where the quarters dont match, return 0 if None and 1 otherwise

        for req in self.profile.requests: 
            if req.quarters == []: 
                notAllowedQuarters = []
                allowedQuarters = self.profile.quarters
            else: 
                notAllowedQuarters = quarters - set(req.quarters)
                allowedQuarters = req.quarters
            #print 'quarters', quarters, 'req.quarters', req.quarters, 'notAllowedQuarters', notAllowedQuarters
            for quarter in notAllowedQuarters: 
                #print 'adding unary potential({0}. {1}) = quarters_match'.format(req,quarter)
                csp.add_unary_potential((req, quarter), quarters_dont_match)
            for quarter in allowedQuarters: 
                csp.add_unary_potential((req, quarter), quarters_match)
        
        # END_YOUR_CODE

    def add_request_weights(self, csp):
        """
        Incorporate weights into the CSP. By default, a request has a weight
        value of 1 (already configured in Request). You should only use the
        weight when one of the requested course is in the solution. A
        unsatisfied request should also have a weight value of 1.

        @param csp: The CSP where the additional constraints will be added to.
        """
        # Problem 3b
        # BEGIN_YOUR_CODE (around 5 lines of code expected)
        # foreach request and quarter, return its weight
        for req in self.profile.requests: 
            for quarter in self.profile.quarters: 
                def weight(req_arg): 
                    if req_arg is None: 
                        return 1
                    return req.weight
                csp.add_unary_potential((req, quarter),weight)
        # END_YOUR_CODE


    def add_prereq_constraints(self, csp):
        """
        Adding constraints to enforce prerequisite. A course can have multiple
        prerequisites. You can assume that *all courses in req.prereqs are
        being requested*. Note that if our parser inferred that one of your
        requested course has additional prerequisites that are also being
        requested, these courses will be added to req.prereqs. You will be notified
        with a message when this happens. Also note that req.prereqs apply to every
        single course in req.cids. If a course C has prerequisite A that is requested
        together with another course B (i.e. a request of 'A or B'), then taking B does
        not count as satisfying the prerequisite of C. You cannot take a course
        in a quarter unless all of its prerequisites have been taken *before* that
        quarter. You should take advantage of get_or_variable().

        @param csp: The CSP where the additional constraints will be added to.
        """
        # Iterate over all request courses
        for req in self.profile.requests:
            if len(req.prereqs) == 0: continue
            # Iterate over all possible quarters
            for quarter_i, quarter in enumerate(self.profile.quarters):
                # Iterate over all prerequisites of this request
                for pre_cid in req.prereqs:
                    # Find the request with this prerequisite
                    for pre_req in self.profile.requests:
                        if pre_cid not in pre_req.cids: continue
                        # Make sure this prerequisite is taken before the requested course(s)
                        prereq_vars = [(pre_req, q) \
                            for i, q in enumerate(self.profile.quarters) if i < quarter_i]
                        v = (req, quarter)
                        orVar = get_or_variable(csp, (v, pre_cid), prereq_vars, pre_cid)
                        # Note this constraint is enforced only when the course is taken
                        # in `quarter` (that's why we test `not val`)
                        csp.add_binary_potential(orVar, v, lambda o, val: not val or o)

    def add_unit_constraints(self, csp):
        """
        Add constraint to the CSP to ensure that the total number of units are
        within profile.minUnits/maxmaxUnits, inclusively. The allowed range for
        each course can be obtained from bulletin.courses[cid].minUnits/maxUnits.
        For a request 'A or B', if you choose to take A, then you must use a unit
        number that's within the range of A. You should introduce any additional
        variables that you are needed. In order for our solution extractor to
        obtain the number of units, for every course that you plan to take in
        the solution, you must have a variable named (courseId, quarter) (e.g.
        ('CS221', 'Aut2013') and it's assigned value is the number of units.
        You should take advantage of get_sum_variable().

        @param csp: The CSP where the additional constraints will be added to.
        """
        # Problem 3c
        # Hint 1: read the documentation above carefully
        # Hint 2: the domain for each (courseId, quarter) variable should contain 0
        #         because the course might not be taken
        # Hint 3: use nested functions and lambdas like what get_or_variable and
        #         add_prereq_constraints do
        # Hint 4: don't worry about quarter constraints in each Request as they'll
        #         be enforced by the constraints added by add_quarter_constraints

        def sum_is_ok(sum): 
            return sum >= self.profile.minUnits and sum <= self.profile.maxUnits
        # add all variables (course, quarter), allowedUnits
        #print self.profile.quarters
        for quarter in self.profile.quarters: 
            variables = []           
            for req in self.profile.requests: 
                for course in req.cids: 
                    domain = [0]
                    for unit in xrange(self.bulletin.courses[course].minUnits, self.bulletin.courses[course].maxUnits + 1): 
                        domain.append(unit)
                    #print 'domain for {0}: {1}'.format(self.bulletin.courses[course].__str__(), domain)
                    var = (course, quarter)
                    #print 'csp.add_variable({0}, {1})'.format(var, domain)
                    csp.add_variable(var, domain) # PROBLEM, but requried by asst
                    variables.append(var)
                    # make sure that each course is listed with the number of units it should have
                    def quarterAlignment(req_arg, unit): 
                        if req_arg == course:
                            return int(unit > 0)
                        return int(unit == 0)
                    #print 'add_binary_potential(({0}, {1}), {2}, quarterAlignment)'.format(req, quarter, var)
                    csp.add_binary_potential((req, quarter), var, quarterAlignment)
            #print 'calling get_sum_variable({0},{1})'.format(variables, self.profile.maxUnits)
            sumVar = get_sum_variable(csp, 'unitSum'+ quarter, variables,  self.profile.maxUnits)
            csp.add_unary_potential(sumVar, sum_is_ok)


    def add_all_additional_constraints(self, csp):
        """
        Add all additional constraints to the CSP.

        @param csp: The CSP where the additional constraints will be added to.
        """
        self.add_quarter_constraints(csp)
        self.add_request_weights(csp)
        self.add_prereq_constraints(csp)
        self.add_unit_constraints(csp)
