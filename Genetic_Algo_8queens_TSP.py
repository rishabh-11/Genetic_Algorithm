import random
from matplotlib import pyplot as plt
print("Assignment 1 AI")
print("Please enter 1 for solving 8-queens problem and 2 for solving TSP")
x = input()
if x!="1" and x!="2":
    print("Invalid input")
    exit(0)
elif x=="1":
    # fitness function
    def find_fitness(values):
        ans=0
        size = len(values)
        for i in range(size):
            for j in range(i+1,size):
                # if queens lie in the same row or are at the same diagonal increment answer
                if values[i]==values[j] or (abs(values[i]-values[j])==abs(i-j)): 
                    ans+=1
        return 29-ans
    # reproduce function
    def reproduce(x,y):
        z = random.randint(1,7)
        #stochastically calculate new state from the two possible reproduced states 
        w = x.values[0:z] + y.values[z:]
        t = y.values[0:z] + x.values[z:]
        temp = [w,t]
        temp1 = [find_fitness(w),find_fitness(t)]
        f = random.choices(temp,temp1)[0]
        new_state1 = state(f)
        return new_state1
    
    # stochastically selecting a state from the current population based on fitness value
    def random_selection(population,weights):
        selected_state = random.choices(population,weights)
        return selected_state[0]

    # function for finding the maximum fitness in the population
    def max_fitness(population):
        ans = 1
        for i in population:
            ans = max(ans,i.fitness)
        return ans

    # list for initializing the population
    list = []
    for i in range(0,8):
        list.append(1)

    # a class which represents the states of the 8 queens problem
    # values is the list of positions where the queens are present in the columns
    # fitness represents the fitness value of the state
    class state:
        #constructor of the class to initialize the values
        def __init__(self,a):
            self.values = a
            self.fitness = find_fitness(self.values)
        #mutate function for causing mutations in the state
        def mutate(self):
            #list for storing all the possible attacking pairs
            list = []
            for i in range(len(self.values)):
                for j in range(i+1,len(self.values)):
                    if self.values[i]==self.values[j] or (abs(self.values[i]-self.values[j])==abs(i-j)):
                        list.append([i,j])
            # if list is empty then the reproduced state is the solution and no mutatation is needed
            if len(list)==0:
                return
            #choose any pair at random and then choose any one of the index for mutation
            index = random.choice(list)
            index2 = random.choice([0,1])
            a = -1
            if index2==0:
                a = index[0]
            else:
                a = index[1]
            b = random.choice([i for i in range(1,9) if i not in [self.values[a]]])
            self.values[a] = b
            self.fitness = find_fitness(self.values)
    
    no_of_generations = 5000 #number of generations the algorithm will run for
    population_size = 50 #population size for each generation

    #initialization of population
    population = []
    for i in range(0,population_size):
        population.append(state(list))

    prev_fitness = 1 # for keeping track of fitness value of previous generation
    c = 70 # number of generations for which the value of maximum fitness can stay constant
    count = no_of_generations
    mut_prob = 0.98 #mutation probability
    ans = 1 # maximum fitness obtained after running the algorithm
    best_fitness = [] #list of maximum fitness for each generation
    res = [] # resultant output list of the algorithm
    current_fitness = []  # list for maintaining current fitness of the population

    # implementation of the algorithm
    for i in range(0,population_size):
        current_fitness.append(1)
    while(count>0 and ans!=29):
        new_population = [] # new population generated from the previous one
        new_fitness = [] # fitness values of the new population 
        for i in range(0,len(population)):
            x = random_selection(population,current_fitness)
            y = random_selection(population,current_fitness)
            temp = reproduce(x,y)
            if random.uniform(0,1)>=mut_prob:
                temp.mutate()
            if mut_prob==0:
                mut_prob = 0.98 #reset the mutation probability after mutation
            new_population.append(temp)
            new_fitness.append(1.5**temp.fitness) #giving more weight to higher fitness values by exponentiating
            if ans<temp.fitness:
                ans = temp.fitness
                res = temp.values
            if ans==29:
                break
        best_fitness.append(max_fitness(new_population))
        if(max_fitness(new_population)==prev_fitness): # checking if max fitness is same as the previous generation
            if(c>0):
                c-=1
            else:
                mut_prob = 0 # causing mutation with 100% probability
        else:
            c=70
        prev_fitness = max_fitness(new_population)
        population = new_population
        current_fitness = new_fitness
        count-=1
    print(no_of_generations - count)
    print(ans)
    print(res)
    plt.xlabel("generations")
    plt.ylabel("best_fitness_value")
    plt.plot(best_fitness)
    plt.title("Best fitness value of each generation")
    plt.show()

elif x=="2":
    # cities are represented by numbers 0,1,.....,13
    #distance between cities
    map = [[0,1000,1000,1000,1000,1000,0.15,1000,1000,0.2,1000,0.12,1000,1000],
       [1000,0,1000,1000,1000,1000,1000,0.19,0.4,1000,1000,1000,1000,0.13],
       [1000,1000,0,0.6,0.22,0.4,1000,1000,0.2,1000,1000,1000,1000,1000],
       [1000,1000,0.6,0,1000,0.21,1000,1000,1000,1000,0.3,1000,1000,1000],
       [1000,1000,0.22,1000,0,1000,1000,1000,0.18,1000,1000,1000,1000,1000],
       [1000,1000,0.4,0.21,1000,0,1000,1000,1000,1000,0.37,0.6,0.26,0.9],
       [0.15,1000,1000,1000,1000,1000,0,1000,1000,1000,0.55,0.18,1000,1000],
       [1000,0.19,1000,1000,1000,1000,1000,0,1000,0.56,1000,1000,1000,0.17],
       [1000,0.4,0.2,1000,0.18,1000,1000,1000,0,1000,1000,1000,1000,0.6],
       [0.2,1000,1000,1000,1000,1000,1000,0.56,1000,0,1000,0.16,1000,0.5],
       [1000,1000,1000,0.3,1000,0.37,0.55,1000,1000,1000,0,1000,0.24,1000],
       [0.12,1000,1000,1000,1000,0.6,0.18,1000,1000,0.16,1000,0,0.4,1000],
       [1000,1000,1000,1000,1000,0.26,1000,1000,1000,1000,0.24,0.4,0,1000],
       [1000,0.13,1000,1000,1000,0.9,1000,0.17,0.6,0.5,1000,1000,1000,0]]

    # fitness function 
    def find_fitness(list):
        # fitness value is defined as the inverse of the length of the path corresponding to the state
        fitness_value=0
        for i in range(len(list)-1):
            fitness_value+=map[list[i]][list[i+1]]
        fitness_value+=map[list[len(list)-1]][list[0]]
        return 1/fitness_value

    # reproduce function
    def reproduce(x,y):
        # find the subset that is to be taken from the first list x
        z = random.randint(0,13)
        w = random.randint(0,13)
        # add the elements of the subset in the list at their appropriate positions
        new_list = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
        for j in range (min(z,w),max(z,w)+1):
            new_list[j]=x[j]
        # add the remaining elements from the list y while maintaing the order of elements in it
        k=0
        for j in range (14):
            if(y[j] not in new_list):
                while(new_list[k]!=-1):
                    k+=1
                new_list[k]=y[j]
                k+=1
        new_state = state(new_list)
        return new_state

    # stochastically selecting a state from the current population based on fitness value
    def random_selection(population,weights):
        selected_state = random.choices(population,weights)
        return selected_state[0]

    # function for finding the maximum fitness in the population
    def max_fitness(population):
        ans = 0
        for i in population:
            ans = max(ans,i.fitness)
        return ans

    # list for initializing the population
    list = []
    for i in range(14):
        list.append(i)

    # a class which represents the states of the TSP
    # values represent the order in which the cities are travelled starting from city at index 0 of the list.
    # fitness represents the fitness value of the state
    class state:
        #constructor for initializing values
        def __init__(self,a):
            self.values = a
            self.fitness = find_fitness(self.values)
        #mutation functions
        #here we are using 2 mutation functions for improving the algorithm
        def mutate(self):
            # randomly select 2 cities and interchange their position in the permutation
            a = random.randint(0,13)
            b = random.choice([i for i in range(0,14) if i not in [a]])
            temp1 = self.values[a]
            self.values[a] = self.values[b]
            self.values[b] = temp1
            self.fitness = find_fitness(self.values)
        
        #second mutation which will be applied immediately after the first one.
        def mutate1(self):
            # Find two cities with minimum distance between them such that the edge connecting them is not part of the original tour 
            # initializing the values for finding the 2 cities
            city_val1 = -1 
            city_val2 = -1
            # val represents the distance between the cities
            val = 50000
            
            # not_to_take represents the list of edges that cant be taken because they are part of the original tour
            not_to_take = [(self.values[len(self.values)-1],self.values[0]),(self.values[0],self.values[len(self.values)-1])]
            for i in range(len(self.values)-1):
                not_to_take.append((self.values[i],self.values[i+1]))
                not_to_take.append((self.values[i+1],self.values[i]))
            
            # finding city_val1 and city_val2
            for i in range(len(map)):
                for j in range(len(map)):
                    if i==j:
                        continue
                    if val>map[i][j] and (i,j) not in not_to_take:
                        city_val1 = i
                        city_val2 = j
                        val = map[i][j]
            
            #making sure that the index of city_val1 is before index of city_val2 in the original list to reduce the number of comparisions
            # in the coming steps   
            if self.values.index(city_val1)>self.values.index(city_val2):
                city_val2,city_val1=city_val1,city_val2
                
            #now compare the edges with city_val1 and city_val2 as vertices in the original tour and pick the one with max distance
            # this edge will be removed from the tour and the edge between city_val1 and city_val2 will be added to the tour.
            # this will be achieved by swapping two cities accordingly
    
            city_index3 = -1 # represents the index of the neighbouring city of city_val1 in the original tour with maximum distance
            city_index4 = -1 # represents the index of the neighbouring city of city_val2 in the original tour with maximum distance
            flag = 0 #tells us which city has to be swapped city_val1 or city_val2
            val = 0 # represents the maximum value of among the neighbouring distances of city_val1 and city_val2

            if self.values.index(city_val1)==0:
                if val<map[city_val1][self.values[self.values.index(city_val1)+1]]:
                    val = map[city_val1][self.values[self.values.index(city_val1)+1]]
                    city_index3 = 1
                    flag = 1
                if val<map[city_val1][self.values[len(self.values)-1]]:
                    val = map[city_val1][self.values[len(self.values)-1]]
                    city_index3 = len(self.values)-1
                    flag = 1
            else:
                if val<map[city_val1][self.values[self.values.index(city_val1)+1]]:
                    val = map[city_val1][self.values[self.values.index(city_val1)+1]]
                    city_index3 = self.values.index(city_val1)+1
                    flag = 1
                if val<map[city_val1][self.values[self.values.index(city_val1)-1]]:
                    val = map[city_val1][self.values[self.values.index(city_val1)-1]]
                    city_index3 = self.values.index(city_val1)-1
                    flag = 1
            
            if self.values.index(city_val2)==len(self.values)-1:
                if val<map[city_val2][self.values[0]]:
                    val = map[city_val2][self.values[0]]
                    city_index4 = 0
                    flag = 0
                if val<map[city_val2][self.values[len(self.values)-2]]:
                    val = map[city_val2][self.values[len(self.values)-2]]
                    city_index4 = len(self.values)-2
                    flag = 0
            else:
                if val<map[city_val2][self.values[self.values.index(city_val2)+1]]:
                    val = map[city_val2][self.values[self.values.index(city_val2)+1]]
                    city_index4 = self.values.index(city_val2)+1
                    flag = 0
                if val<map[city_val2][self.values[self.values.index(city_val2)-1]]:
                    val = map[city_val2][self.values[self.values.index(city_val2)-1]]
                    city_index4 = self.values.index(city_val2)-1
                    flag = 0
            # if flag is 1 then city_val1 has highest neighbouring distance and therefore we swap city_val2 with the city corresponding to city_index3
            # if flag is 0 then city_val2 has highest neighbouring distance and therefore we swap city_val1 with the city corresponding to city_index4
            if flag==1:
                self.values[city_index3],self.values[self.values.index(city_val2)]=self.values[self.values.index(city_val2)],self.values[city_index3]
            else:
                self.values[city_index4],self.values[self.values.index(city_val1)]=self.values[self.values.index(city_val1)],self.values[city_index4]
            
    population_size = 50 # size of the population
    no_of_generations = 1000 # number of generations for which the algorithm will run

    #initialize the population
    population = []
    for i in range(0,population_size):
        population.append(state(list))

    count = no_of_generations
    ans_overall = 0 #represents the maximum fitness obtained by the algorithm
    mut_prob = 0.98 #mutation probability
    best_fitness = [] #maximum fitness value for each generation
    final_list = [] #final list which represents the order of traversal for maximum fitness obtaonrd by the algorithm
    current_fitness = [] # list for storing the current fitness of the population

    # implementation of the genetic algorithm
    for i in range(0,population_size):
        current_fitness.append(1)
    while(count>0):
        new_population = [] # new population generated from the previous one
        new_fitness = []    # fitness values of the new population 
        for i in range(0,len(population)):
            x = random_selection(population,current_fitness) 
            y = random_selection(population,current_fitness)
            temp = reproduce(x.values,y.values)
            if random.uniform(0,1)>=mut_prob:
                # first call mutate and then mutate1
                temp.mutate()
                temp.mutate1()
            new_population.append(temp)
            new_fitness.append(temp.fitness)
            if ans_overall<temp.fitness:
                ans_overall=temp.fitness
                final_list = temp.values
        best_fitness.append(max_fitness(new_population))
        population = new_population
        current_fitness = new_fitness
        count-=1
    print(1/ans_overall)
    print(final_list)
    plt.xlabel("generations")
    plt.ylabel("best_fitness_value")
    plt.plot(best_fitness)
    plt.title("Best fitness value of each generation")
    plt.show()

