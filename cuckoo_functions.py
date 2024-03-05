import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import gamma as gamma_function
import math
import time

# Cuckoo Search with the main function
class Cuckoo():
    
    # Constants

    Lower_Bound = 0            # Test Area
    Upper_Bound = 100          # Test Area
    Min_Alp = 0.9      # Step Size 
    Max_Alp = 1.0      # Step Size
    PA_M_Minimum = 0.05        # Mutation Prob
    PA_M_Maximum = 0.25        # Mutation Prob
    Nest_A = 25          # Test Field Sols.(TOTAL)
    Total_Iterations = 100    # For every node
    gamma = 0.1          # Noise
    CONST_Levy = 1.5          # Levy flight const.
    Node_Density = 0      
    Anchor_Ratio = 0   
    R = 0          # Coverage area of each Node
    
    Anchors = []        
    Undecided_Nodes = []      
    Location = []          
    Minimum_Fitness = []          
    
    # Initializing the values
    def __init__(self, Node_Density = 100, Anchor_Ratio = 0.20, R = 25):
        self.Node_Density = Node_Density
        self.Anchor_Ratio = Anchor_Ratio
        self.R = R
        self.anc = int(self.Anchor_Ratio * self.Node_Density)        # Calculation of number of anchor nodes
        self.unknowns = self.Node_Density - self.anc                   # Calculation of number of unknown nodes        
        
        # Placing Anchors Randomly in Test Field
        for i in range(self.anc):
            Coord_X = np.random.randint(self.Upper_Bound)
            Coord_Y = np.random.randint(self.Upper_Bound)
            self.Anchors.append([Coord_X, Coord_Y])
        
        # Placing Unknown Nodes Ranndomly in Test Field
        for i in range(self.unknowns):
            x_val = np.random.randint(self.Upper_Bound)
            y_val = np.random.randint(self.Upper_Bound)
            self.Undecided_Nodes.append([x_val, y_val])
        
        self.Undecided_Nodes = np.array(self.Undecided_Nodes)
        self.Anchors = np.array(self.Anchors)
    
        # Store Original Coords (Anchor & Unknown)
        self.Anchors_og = self.Anchors.copy()
        self.Undecided_Nodes_og = self.Undecided_Nodes.copy()
    
    
    # plotting the test field
    def plot_graph(self, unknown = True):
        plt.figure(figsize=(8, 8))
        if unknown == True:
            if len(self.Undecided_Nodes) != 0:
                plt.plot(self.Undecided_Nodes[:, 0], self.Undecided_Nodes[:, 1], 'yo', label="Left Node")
            plt.plot(self.Undecided_Nodes_og[:, 0], self.Undecided_Nodes_og[:, 1], 'ro', label='Unknown Node')
        else:
            plt.plot(self.Undecided_Nodes_og[:, 0], self.Undecided_Nodes_og[:, 1], 'ro', label='Unknown Node')
            if len(self.Undecided_Nodes) != 0:
                plt.plot(self.Undecided_Nodes[:, 0], self.Undecided_Nodes[:, 1], 'yo', label="Left Node")
        plt.plot(self.Anchors_og[:, 0], self.Anchors_og[:, 1], 'go', label="Anchor Node")
        plt.plot(self.Anchors[self.anc:, 0], self.Anchors[self.anc:, 1], 'b^', label="Localized Node", fillstyle='none', markersize=10)
        
        plt.legend()
        plt.axis([0,self.Upper_Bound,0,self.Upper_Bound])
        plt.grid()
        plt.show()
    
    
    def Alp(self, iteration): # Levy Flight Step Size
        return self.Max_Alp - ((iteration/ self.Total_Iterations) * (self.Max_Alp - self.Min_Alp)) 

    def levy_flight(self): #Levy Flight walk
        Temp_Var = np.power(((gamma_function(1 + self.CONST_Levy) * np.sin(np.pi * (self.CONST_Levy /2))) / (gamma_function((1 + self.CONST_Levy)/2) * self.CONST_Levy * np.power(2, ((self.CONST_Levy - 1)/2)) )), 1/self.CONST_Levy)
        Random_Var_1 = np.random.normal(0, Temp_Var)
        Random_Var_2 = np.random.normal(0,1)
        Walk_Value = Random_Var_1 / (np.power(abs(Random_Var_2), (1/self.CONST_Levy)))

        return Walk_Value 

    
    def Limit_Nodes(self, point): # Nodes cant leave Test Field      
      a = point[0]
      b = point[1]
      if a > self.Upper_Bound and b > self.Upper_Bound: 
         a,b = self.Upper_Bound, self.Upper_Bound
          

         a,b = self.Upper_Bound, b
      elif a > self.Upper_Bound and b < self.Lower_Bound:
          a,b = self.Upper_Bound, self.Lower_Bound
      elif self.Lower_Bound < a < self.Upper_Bound and b < self.Lower_Bound:
          a,b = a, self.Lower_Bound
      elif a < self.Lower_Bound and b < self.Lower_Bound:
          a,b = self.Lower_Bound, self.Lower_Bound
      elif a < self.Lower_Bound and self.Lower_Bound < b < self.Upper_Bound:
          a,b = self.Lower_Bound, b
      elif a < self.Lower_Bound and b > self.Upper_Bound:
          a,b = self.Lower_Bound, self.Upper_Bound
      elif self.Lower_Bound < a < self.Upper_Bound and b > self.Upper_Bound:
          a,b = a, self.Upper_Bound
      
      return [a,b]


    def Find_Neighbours(self, node, anchors): # Neighbours(Anchor Nodes) within Range of Transmission
     a = node[0]
     b = node[1]
     Anchor_Node_List = anchors
     Coord_Neighbouring_Anchors = []
     for j in range(len(Anchor_Node_List)): 
         Distance = np.power((np.power((a - Anchor_Node_List[j][0]), 2) + np.power((b - Anchor_Node_List[j][1]), 2)), 0.5)
         np.random.seed(2)
         Distance_Error = Distance + np.random.normal(0, (self.gamma*Distance))
         np.random.seed()

        
         if Distance_Error < self.R: # Checking Vicinity
             Coord_Neighbouring_Anchors.append(Anchor_Node_List[j])
   
     return Coord_Neighbouring_Anchors 

    def objective_function(self, node, Anchor_Nodes, Unknown_Nodes, Nodes_Minimum = 3):
       a = node[0]
       b = node[1]

       a1 = Unknown_Nodes[0]
       b1 = Unknown_Nodes[1]
    
       Neighbours_of_Unknown = self.Find_Neighbours(Unknown_Nodes, Anchor_Nodes) # Finding the Neighbours of the Unknown Node
       Number_of_Neighbours = len(Neighbours_of_Unknown)
       Ranging_Error = []
       if len(Neighbours_of_Unknown) >= Nodes_Minimum:
          for ancn in Neighbours_of_Unknown: 
              # Distance b/w approximated Node and Anchor
              Distance = np.power((np.power((a - ancn[0]), 2) + np.power((b - ancn[1]), 2)), 0.5)
                
              # Distance b/w Unknown Node and Anchor
              Distance_U_A = np.power((np.power((a1 - ancn[0]), 2) + np.power((b1 - ancn[1]), 2)), 0.5)
                
              # Distance including the Ranging Error
              np.random.seed(2)
              Distance_Error = Distance_U_A + np.random.normal(0, (self.gamma*Distance_U_A))
              np.random.seed()
              Ranging_Error.append(np.power(Distance - Distance_Error,2))

         # Not enough anchors in range means infinite value
          ans = None
          if math.isnan(np.sum(Ranging_Error)/Number_of_Neighbours): 
              ans = np.inf
          else:
              ans = np.sum(Ranging_Error)/Number_of_Neighbours or None
            
          return ans # MSE (Ranging Error)

    
    def Enhanced_Cuckoo_Search(self, Anchors_A, Undecided_Node):
      
        # Population initialisation
        Nest_B = []                   
        Minimum_Number_Of_Nodes = len(Anchors_A)     
        for i in range(self.Nest_A):
            nest_a = np.random.randint(self.Upper_Bound)
            nest_b = np.random.randint(self.Upper_Bound)
            Nest_B.append([nest_a, nest_b])
        
        # Fitness calculation
        Objective_val_Nest_B = []              
        for i in range(len(Nest_B)):
            Objective_val_Nest_B.append(self.objective_function(Nest_B[i], Anchors_A, Undecided_Node, Minimum_Number_Of_Nodes) or np.inf)

        
        Iteration = 0                   
        Fitness_Minimum = []                   
        while(Iteration < self.Total_Iterations):
            Iteration += 1              
            New_Solutions = []                
            Fitness_New_solutions = []            
            # Levy flight to each solution
            for i in range(len(Nest_B)):
                X = Nest_B[i].copy()
                X[0] = X[0] + self.Alp(Iteration) * self.levy_flight()
                X[1] = X[1] + self.Alp(Iteration) * self.levy_flight()
                
                X = self.Limit_Nodes(X)     
                
                Fitness_X = self.objective_function(X, Anchors_A, Undecided_Node, Minimum_Number_Of_Nodes) or np.inf
                Random_z = np.random.randint(0, len(Nest_B))
                Fitness_Y = Objective_val_Nest_B[Random_z]
                
                # New sol and Old sol Fitness comparision
                if Fitness_X > Fitness_Y:
                    X[0] = Nest_B[Random_z][0]
                    X[1] = Nest_B[Random_z][1]
                    Fitness_X = Fitness_Y

                New_Solutions.append(X)
                Fitness_New_solutions.append(Fitness_X)
            
            # Best Current Solution
            Fitness_New_solutions = np.array([np.inf if i is None else i for i in Fitness_New_solutions])
            Fitness_Minimum_New = Fitness_New_solutions[np.argmin(Fitness_New_solutions)]

            Best_Solution_Current = New_Solutions[np.argmin(Fitness_New_solutions)]
            
            # Mutation Probability (PA_M)
            PA_M = []                 
            for i in Fitness_New_solutions:
                K = i - Fitness_Minimum_New
                if K < 1:
                    PA_M.append(self.PA_M_Minimum + (self.PA_M_Maximum - self.PA_M_Minimum) * K)
                else:
                   PA_M.append(self.PA_M_Maximum / Iteration)
  
            # Mutating
            for i in range(len(PA_M)):
                Temp_Var = np.random.uniform(0, 1)
                if Temp_Var < PA_M[i]:
                    x = np.random.randint(self.Upper_Bound)
                    y = np.random.randint(self.Upper_Bound)
                    Fitness_y_x = self.objective_function([x, y], Anchors_A, Undecided_Node, Minimum_Number_Of_Nodes) or np.inf
                    if (Fitness_New_solutions[i]) > (Fitness_y_x):
                        New_Solutions[i] = [x, y]
                        Fitness_New_solutions[i] = Fitness_y_x
          
            # Best Current Solution
            Fitness_New_solutions = np.array([np.inf if i is None else i for i in Fitness_New_solutions])
            Fitness_Minimum_New = Fitness_New_solutions[np.argmin(Fitness_New_solutions)]

            self.Minimum_Fitness.append(Fitness_Minimum_New)
            Fitness_Minimum.append(Fitness_Minimum_New)
              
            # Criteria/Mechanism for stopping (enhanced)
            if len(Fitness_Minimum) > 3:
                a = abs(Fitness_Minimum[-1] - Fitness_Minimum[-2])
                b = abs(Fitness_Minimum[-2] - Fitness_Minimum[-3])
                c = abs(Fitness_Minimum[-3] - Fitness_Minimum[-4])
                if a == b == c == 0:
                    break
        
            # Coordinates of Best Solution (Iteration)
            Best_Solution = New_Solutions[np.argmin(Fitness_New_solutions)]
            Nest_B = New_Solutions.copy()
            Objective_val_Nest_B = Fitness_New_solutions.copy()
       
        return Best_Solution 
                
    
    def Update_Undecided(self, indexes): #Unknown is now anchor       
        Undecided_A_temp = []
        for j in range(len(self.Undecided_Nodes)):
            if j in indexes:
                pass
            else:
                Undecided_A_temp.append(self.Undecided_Nodes[j])

        self.Undecided_Nodes = np.array(Undecided_A_temp)
        

    def main(self):       # Node Localization
        for j in range(5):
            Nodes_Localized = []
            for i in range(len(self.Undecided_Nodes)):
                Nearest_Neighbour = self.Find_Neighbours(self.Undecided_Nodes[i], self.Anchors)
                if len(Nearest_Neighbour) >= 3:
                    Node_Location_New = self.Enhanced_Cuckoo_Search(Nearest_Neighbour, self.Undecided_Nodes[i])

                    self.Location.append([Node_Location_New, self.Undecided_Nodes[i]])
              
                    Anchor_Node_Temp = list(self.Anchors)
                    Anchor_Node_Temp.append(Node_Location_New)
                    self.Anchors = np.array(Anchor_Node_Temp)

                    Nodes_Localized.append(i)

            self.Update_Undecided(Nodes_Localized)