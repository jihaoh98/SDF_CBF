


    

 
 
    

        
def solve_beta():
    # robot_vertices = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    # obs_vertices = np.array([[3.0, 1.5], [4.0, 1.5], [4.0, 2.0], [3.0, 2.0]])
    # A1, b1, b0 = get_halfspace_constraints(robot_vertices, np.array([1.0, 0.5]))    
    # A2, b2, _ = get_halfspace_constraints(obs_vertices)

    robot_vertices = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
    
    A1, b1, b0 = get_halfspace_constraints(robot_vertices, np.array([1.5, 1.5]))    
    A2, b2, _ = get_halfspace_constraints(obs_vertices)
    
    # plot_polytope(robot_vertices, obs_vertices)
    # derive_min_distance(A1, b1, A2, b2)




