"""
Knowledge-Based Agent for Hazardous Warehouse using Z3.

Implements Tasks 1-6: Setup, symbols, manual reasoning, agent loop, testing, and reflection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from z3 import Bool, Bools, Or, And, Not, Solver, unsat
from collections import deque
from hazardous_warehouse_env import HazardousWarehouseEnv, AgentAction, Percept
from hazardous_warehouse_viz import configure_rn_example_layout, print_layout, print_percepts


# ============================================================================
# TASK 1: Setup and Exploration
# ============================================================================

def test_z3_basics():
    """Test basic Z3 functionality."""
    print("=" * 70)
    print("TASK 1: Z3 Basics and Entailment")
    print("=" * 70)
    
    # Create boolean variables
    P, Q = Bools('P Q')
    print(f"[OK] Created boolean variables: P, Q")
    
    # Create solver and add constraints
    s = Solver()
    s.add(P == Q)  # Biconditional: P iff Q
    s.add(P)       # Fact: P is true
    print(f"[OK] Added constraints: P==Q and P")
    
    # Check satisfiability
    result = s.check()
    print(f"[OK] Satisfiability check: {result}")
    
    # Inspect the model
    if result.r == 1:  # sat
        model = s.model()
        print(f"[OK] Model: P = {model[P]}, Q = {model[Q]}")
    
    # Test entailment
    print("\nTesting z3_entails function:")
    s2 = Solver()
    s2.add(P == Q)
    s2.add(P)
    entails = z3_entails(s2, Q)
    print(f"[OK] Does KB entail Q? {entails}")
    
    return True


def z3_entails(kb_solver, query):
    """
    Check if knowledge base entails a query using push/pop (Section 3.6).
    
    Args:
        kb_solver: A Z3 Solver with the KB already added
        query: A Z3 expression to check
    
    Returns:
        bool: True if KB entails query, False otherwise
    """
    kb_solver.push()  # Save solver state
    kb_solver.add(Not(query))  # Add negation of query
    
    is_unsat = kb_solver.check() == unsat  # If unsat, KB entails query
    
    kb_solver.pop()  # Restore solver state
    
    return is_unsat


# ============================================================================
# TASK 2: Symbols and Physics
# ============================================================================

def damaged(r, c):
    """Boolean variable: True if cell (r,c) has damaged floor."""
    return Bool(f"D({r},{c})")


def forklift_at(r, c):
    """Boolean variable: True if forklift is at cell (r,c)."""
    return Bool(f"F({r},{c})")


def creaking_at(r, c):
    """Boolean variable: True if creaking percept at cell (r,c)."""
    return Bool(f"Creaking({r},{c})")


def rumbling_at(r, c):
    """Boolean variable: True if rumbling percept at cell (r,c)."""
    return Bool(f"Rumbling({r},{c})")


def safe(r, c):
    """Boolean variable: True if cell (r,c) is safe to visit."""
    return Bool(f"Safe({r},{c})")


def build_warehouse_kb(width=4, height=4):
    """
    Build knowledge base for warehouse with physics rules (Section 3.6).
    
    Rules:
    1. A cell is safe iff no damaged floor and no forklift
    2. Creaking iff damaged floor is adjacent (including diagonals)
    3. Rumbling iff forklift is adjacent (including diagonals)
    
    Args:
        width: Grid width
        height: Grid height
    
    Returns:
        z3.Solver: Configured solver with warehouse physics
    """
    solver = Solver()
    
    # Rule 1: Safe cells have no damage and no forklifts
    for r in range(height):
        for c in range(width):
            s = safe(r, c)
            d = damaged(r, c)
            f = forklift_at(r, c)
            # Safe(r,c) iff not D(r,c) and not F(r,c)
            solver.add(s == And(Not(d), Not(f)))
    
    # Rule 2: Creaking iff adjacent cell has damage (or current cell is damaged)
    for r in range(height):
        for c in range(width):
            creaking = creaking_at(r, c)
            adjacent_damage = []
            
            # Check all adjacent cells (including diagonals)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        adjacent_damage.append(damaged(nr, nc))
            
            if adjacent_damage:
                # Creaking iff any adjacent cell (or self) is damaged
                solver.add(creaking == Or(*adjacent_damage))
            else:
                solver.add(creaking == False)
    
    # Rule 3: Rumbling iff adjacent cell has forklift (or current cell has forklift)
    for r in range(height):
        for c in range(width):
            rumbling = rumbling_at(r, c)
            adjacent_forklifts = []
            
            # Check all adjacent cells (including diagonals)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        adjacent_forklifts.append(forklift_at(nr, nc))
            
            if adjacent_forklifts:
                # Rumbling iff any adjacent cell (or self) has forklift
                solver.add(rumbling == Or(*adjacent_forklifts))
            else:
                solver.add(rumbling == False)
    
    return solver


def test_kb_satisfiability():
    """Test that the knowledge base is satisfiable."""
    print("\n" + "=" * 70)
    print("TASK 2: Knowledge Base Symbols and Physics")
    print("=" * 70)
    
    solver = build_warehouse_kb(4, 4)
    print(f"[OK] Built warehouse KB with physics rules")
    
    result = solver.check()
    print(f"[OK] KB satisfiability: {result}")
    
    if result.r == 1:  # sat
        print(f"[OK] KB is satisfiable")
    
    return solver


# ============================================================================
# TASK 3: Manual Reasoning
# ============================================================================

def tell_solver(solver, r, c, creaking, rumbling):
    """Add percept observations to the solver."""
    solver.add(creaking_at(r, c) == creaking)
    solver.add(rumbling_at(r, c) == rumbling)


def ask_solver(solver, query):
    """Ask if the solver entails a query."""
    return z3_entails(solver, query)


def test_manual_reasoning():
    """Test manual reasoning on Section 3.2 example."""
    print("\n" + "=" * 70)
    print("TASK 3: Manual Reasoning (Section 3.2 Walkthrough)")
    print("=" * 70)
    
    solver = build_warehouse_kb(4, 4)
    print(f"\nInitial KB built and satisfiable")
    
    # Step 1: Percepts at (1,1): no creaking, no rumbling
    print(f"\n--- Step 1: At (1,1) ---")
    print(f"Percepts: No creaking, No rumbling")
    tell_solver(solver, 1, 1, False, False)
    print(f"[OK] TELL solver: Creaking(1,1)=False, Rumbling(1,1)=False")
    
    # ASK about (2,1) and (1,2)
    result_2_1 = ask_solver(solver, safe(2, 1))
    result_1_2 = ask_solver(solver, safe(1, 2))
    print(f"[OK] ASK: Is (2,1) safe? {result_2_1}")
    print(f"[OK] ASK: Is (1,2) safe? {result_1_2}")
    
    # Step 2: Percepts at (2,1): creaking, no rumbling
    print(f"\n--- Step 2: At (2,1) ---")
    print(f"Percepts: Creaking, No rumbling")
    tell_solver(solver, 2, 1, True, False)
    print(f"[OK] TELL solver: Creaking(2,1)=True, Rumbling(2,1)=False")
    
    # ASK about (3,1), (2,2), OK(3,1)
    result_3_1 = ask_solver(solver, safe(3, 1))
    result_2_2 = ask_solver(solver, safe(2, 2))
    result_d_3_1 = ask_solver(solver, damaged(3, 1))
    
    print(f"[OK] ASK: Is (3,1) safe? {result_3_1}")
    print(f"[OK] ASK: Is (2,2) safe? {result_2_2}")
    print(f"[OK] ASK: Is (3,1) damaged? {result_d_3_1}")
    print(f"  => Creaking at (2,1) suggests adjacent damage. Damaged floor likely at (1,0), (1,1), (1,2), (2,0), or (2,2)")
    
    # Step 3: Percepts at (1,2): rumbling, no creaking
    print(f"\n--- Step 3: At (1,2) ---")
    print(f"Percepts: Rumbling, No creaking")
    tell_solver(solver, 1, 2, False, True)
    print(f"[OK] TELL solver: Creaking(1,2)=False, Rumbling(1,2)=True")
    
    # ASK again
    result_3_1_v2 = ask_solver(solver, safe(3, 1))
    result_2_2_v2 = ask_solver(solver, safe(2, 2))
    result_1_3 = ask_solver(solver, safe(1, 3))
    result_d_2_0 = ask_solver(solver, damaged(2, 0))
    result_f_0_2 = ask_solver(solver, forklift_at(0, 2))
    
    print(f"[OK] ASK: Is (3,1) safe? {result_3_1_v2}")
    print(f"[OK] ASK: Is (2,2) safe? {result_2_2_v2}")
    print(f"[OK] ASK: Is (1,3) safe? {result_1_3}")
    print(f"[OK] ASK: Is (2,0) damaged? {result_d_2_0}")
    print(f"[OK] ASK: Is forklift at (0,2)? {result_f_0_2}")
    print(f"  => Rumbling at (1,2) suggests adj forklift. No creaking at (1,2) rules out damage there.")
    
    return solver


# ============================================================================
# TASK 4: Agent Loop
# ============================================================================

class WarehouseKBAgent:
    """
    Knowledge-based agent for hazardous warehouse (Section 3.6).
    
    Cycle: perceive -> tell -> ask -> act
    """
    
    def __init__(self, width=4, height=4):
        """Initialize the KB agent."""
        self.width = width
        self.height = height
        self.solver = build_warehouse_kb(width, height)
        self.position = (0, 0)
        self.has_package = False
        self.visited = set()
        self.visited.add(self.position)
        self.actions_taken = []
        self.step_count = 0
        
    def perceive(self, environment):
        """Perceive the current state."""
        percept = environment.perceive()
        self.position = environment.agent_pos
        self.has_package = environment.has_package
        
        r, c = self.position
        self.tell(r, c, percept.creaking, percept.rumbling, percept.glitter)
        
        return percept
    
    def tell(self, r, c, creaking, rumbling, glitter):
        """Add percepts to knowledge base."""
        tell_solver(self.solver, r, c, creaking, rumbling)
        
        if glitter:
            # Found the package!
            pass
    
    def ask(self, query):
        """Query the knowledge base."""
        return ask_solver(self.solver, query)
    
    def get_safe_neighbors(self, pos):
        """Get safe neighboring positions."""
        r, c = pos
        safe_neighbors = []
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                # Check if the KB entails this position is safe
                if self.ask(safe(nr, nc)):
                    safe_neighbors.append((nr, nc))
        
        return safe_neighbors
    
    def plan_path_to_target(self, target):
        """
        Use BFS to find a path to target through safe cells.
        
        Returns:
            list: Path of positions, or empty list if no path found
        """
        if self.position == target:
            return [self.position]
        
        queue = deque([(self.position, [self.position])])
        visited = {self.position}
        
        while queue:
            pos, path = queue.popleft()
            r, c = pos
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                next_pos = (nr, nc)
                
                if next_pos in visited:
                    continue
                if not (0 <= nr < self.height and 0 <= nc < self.width):
                    continue
                
                # Check if safe according to KB
                if self.ask(safe(nr, nc)):
                    new_path = path + [next_pos]
                    
                    if next_pos == target:
                        return new_path
                    
                    visited.add(next_pos)
                    queue.append((next_pos, new_path))
        
        return []  # No path found
    
    def get_frontier_neighbors(self, pos):
        """
        Get neighbors that are either safe or unvisited (exploration frontier).
        Prefer unvisited cells to promote exploration.
        """
        r, c = pos
        frontier = []
        unvisited_safe = []
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            next_pos = (nr, nc)
            
            if 0 <= nr < self.height and 0 <= nc < self.width:
                if self.ask(safe(nr, nc)):
                    if next_pos not in self.visited:
                        unvisited_safe.append(next_pos)
                    frontier.append(next_pos)
        
        # Prefer unvisited safe cells, then any safe cell
        return unvisited_safe if unvisited_safe else frontier
    
    def forklift_in_line_of_sight(self):
        """Check if a forklift is in line of sight (horizontal or vertical)."""
        r, c = self.position
        
        # Check horizontal and vertical directions
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            line_r, line_c = r + dr, c + dc
            while 0 <= line_r < self.height and 0 <= line_c < self.width:
                if self.ask(forklift_at(line_r, line_c)):
                    return (line_r, line_c)
                line_r += dr
                line_c += dc
        
        return None
    
    def get_next_action(self, environment):
        """Determine the next action using improved exploration strategy."""
        r, c = self.position
        
        # Bonus: Check for forklift in line of sight
        forklift_pos = self.forklift_in_line_of_sight()
        if forklift_pos:
            print(f"  [!] BONUS: Forklift detected at {forklift_pos}! Firing shutdown device...")
            return AgentAction.SHUTDOWN
        
        # Check if we're at the package
        percept = environment.perceive()
        if percept.glitter and not self.has_package:
            return AgentAction.GRAB
        
        # If we have the package, go to exit
        if self.has_package:
            path = self.plan_path_to_target((0, 0))
            if path and len(path) > 1:
                next_pos = path[1]
                return self.position_to_action(self.position, next_pos)
            else:
                return AgentAction.EXIT
        
        # Otherwise, explore to find package
        # Strategy 1: Move to unvisited safe neighboring cells
        frontier = self.get_frontier_neighbors(self.position)
        
        if frontier:
            next_pos = frontier[0]
            if next_pos not in self.visited:
                self.visited.add(next_pos)
            return self.position_to_action(self.position, next_pos)
        
        # Strategy 2: If stuck exploring currently known safe area,
        # need to take a calculated risk or find alternative path
        # For now, return None (agent is genuinely stuck)
        return None
    
    @staticmethod
    def position_to_action(current, target):
        """Convert position change to action."""
        dr = target[0] - current[0]
        dc = target[1] - current[1]
        
        if dr < 0:
            return AgentAction.UP
        elif dr > 0:
            return AgentAction.DOWN
        elif dc < 0:
            return AgentAction.LEFT
        elif dc > 0:
            return AgentAction.RIGHT
        
        return None


# ============================================================================
# TASK 5: Testing
# ============================================================================

def test_agent_on_example_layout():
    """Run the agent on the example layout and record results."""
    print("\n" + "=" * 70)
    print("TASK 5: Testing Agent on Example Layout")
    print("=" * 70)
    
    # Create environment
    env = configure_rn_example_layout()
    print(f"\nEnvironment configuration:")
    print(f"  Grid: {env.width}x{env.height}")
    print(f"  Damaged floors: {env.damaged_floors}")
    print(f"  Forklifts: {env.forklifts}")
    print(f"  Package: {env.package_location}")
    print(f"  Exit: {env.exit_location}")
    
    print_layout(env)
    
    # Create and run agent
    agent = WarehouseKBAgent(env.width, env.height)
    print(f"Agent started at {agent.position}")
    
    max_steps = 50
    success = False
    
    while not env.is_done() and env.step_count < max_steps:
        # Perceive
        percept = agent.perceive(env)
        print(f"\n[Step {env.step_count}] At {agent.position}")
        print(f"  Percepts: Creaking={percept.creaking}, Rumbling={percept.rumbling}, Glitter={percept.glitter}")
        
        # Get next action
        action = agent.get_next_action(env)
        
        if action is None:
            print(f"  Agent stuck! No safe moves available.")
            break
        
        print(f"  Action: {action.value}")
        agent.actions_taken.append(action)
        
        # Execute action
        reward = env.execute_action(action)
        print(f"  Reward: {reward}")
        
        if env.is_done():
            if env.has_package and env.agent_pos == env.exit_location:
                success = True
            break
    
    # Print results
    print(f"\n" + "=" * 70)
    print(f"RESULTS:")
    print(f"  Success: {success}")
    print(f"  Steps: {env.step_count}")
    print(f"  Total Reward: {env.total_reward}")
    print(f"  Package Retrieved: {env.has_package}")
    print(f"  Exited: {env.agent_pos == env.exit_location}")
    print(f"=" * 70)
    
    return success, env.step_count, env.total_reward


# ============================================================================
# TASK 6: Reflection
# ============================================================================

def reflection():
    """Provide reflection on agent behavior."""
    print("\n" + "=" * 70)
    print("TASK 6: Reflection")
    print("=" * 70)
    
    reflection_text = """
The knowledge-based agent using Z3 demonstrates both strengths and limitations
in reasoning under uncertainty.

Conservative Behavior: The agent tends to be overly conservative in querying
the KB for safety. Once creaking is perceived at a cell, the agent may label
many adjacent cells as unsafe even if they're actually safe. This is because
the KB cannot distinguish between definite knowledge (something IS damaged) and
mere possibility (something MIGHT be damaged). A human would explore cautiously
while the KB agent avoids entire regions.

Getting Stuck: The agent uses BFS to find paths through provably safe cells,
but if the KB proves too many cells unsafe, it may find no path to the target.
This occurs particularly in narrow passages where multiple percepts converge,
causing the KB to over-constrain the solution space.

Missing Capability: The agent lacks probabilistic reasoning. If we used a
probabilistic knowledge base (e.g., Bayesian networks or Markov Logic), we
could assign degrees of belief to uncertain propositions. This would allow
the agent to take calculated risks - moving to cells that are probably safe
but not provably safe. Additionally, incorporating utility theory would allow
the agent to trade off exploration risk against goal achievement.
"""
    
    print(reflection_text)
    
    return reflection_text


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all tasks."""
    
    # Task 1: Setup and exploration
    test_z3_basics()
    
    # Task 2: Symbols and physics
    test_kb_satisfiability()
    
    # Task 3: Manual reasoning
    test_manual_reasoning()
    
    # Task 5: Testing
    success, steps, reward = test_agent_on_example_layout()
    
    # Task 6: Reflection
    reflection()
    
    # Summary
    print("\n" + "=" * 70)
    print("ALL TASKS COMPLETED")
    print("=" * 70)
    print(f"\nTest Results:")
    print(f"  Agent Success: {success}")
    print(f"  Steps Taken: {steps}")
    print(f"  Total Reward: {reward}")


if __name__ == "__main__":
    main()
