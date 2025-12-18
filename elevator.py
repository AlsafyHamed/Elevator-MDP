import tkinter as tk
from tkinter import ttk
import random
import time
from collections import defaultdict

# ============================================================
# ELEVATOR MDP CLASS
# ============================================================

class ElevatorMDP:
    """Markov Decision Process for elevator control"""

    def __init__(self, num_floors=3, new_request_prob=0.1, discount_factor=0.95,
                 move_cost=-0.5, request_penalty=-1.0, pick_reward=5.0):
        self.num_floors = num_floors
        self.new_request_prob = new_request_prob
        self.discount_factor = discount_factor
        self.move_cost = move_cost
        self.request_penalty = request_penalty  # -1 per pending request
        self.pick_reward = pick_reward

        # State: (current_floor, request_mask)
        # Action: UP, DOWN, STAY, PICK
        self.actions = ['UP', 'DOWN', 'STAY', 'PICK']

    def get_states(self):
        """Generate all possible states"""
        states = []
        for floor in range(1, self.num_floors + 1):
            for mask in range(2 ** self.num_floors):
                states.append((floor, mask))
        return states

    def is_valid_action(self, state, action):
        """Check if action is valid in current state"""
        floor, mask = state
        if action == 'UP':
            return floor < self.num_floors
        elif action == 'DOWN':
            return floor > 1
        elif action == 'PICK':
            return self.has_request(mask, floor)
        return True  # STAY is always valid

    def has_request(self, mask, floor):
        """Check if there's a request at a floor"""
        return (mask >> (floor - 1)) & 1

    def count_requests(self, mask):
        """Count total requests in the mask"""
        return bin(mask).count('1')

    def step(self, state, action, deterministic=False):
        """Execute one step of the MDP"""
        floor, mask = state

        # Calculate penalty based on number of pending requests
        num_requests = self.count_requests(mask)
        reward = num_requests * self.request_penalty  # -1 per request

        # Remove request if picking
        if action == 'PICK' and self.has_request(mask, floor):
            mask &= ~(1 << (floor - 1))
            reward += self.pick_reward  # Add pick reward on top of request penalty
        elif action in ['UP', 'DOWN']:
            # Move cost (energy cost)
            reward += self.move_cost
            # Update floor
            if action == 'UP':
                floor = min(floor + 1, self.num_floors)
            else:
                floor = max(floor - 1, 1)

        # For deterministic mode (planning), don't add requests
        if not deterministic:
            # Generate new requests for simulation
            for f in range(1, self.num_floors + 1):
                if random.random() < self.new_request_prob:
                    mask |= (1 << (f - 1))

        next_state = (floor, mask)
        return next_state, reward

    def get_transitions(self, state, action):
        """Get possible transitions - simplified for faster convergence"""
        floor, mask = state

        # Calculate penalty based on number of pending requests
        num_requests = self.count_requests(mask)
        reward = num_requests * self.request_penalty  # -1 per request
        new_mask = mask  # Initialize default

        # Apply action and get immediate reward
        if action == 'PICK' and self.has_request(mask, floor):
            new_mask = mask & ~(1 << (floor - 1))
            reward += self.pick_reward  # Add pick reward
        elif action == 'UP' and floor < self.num_floors:
            floor = floor + 1
            reward += self.move_cost  # Add move cost
        elif action == 'DOWN' and floor > 1:
            floor = floor - 1
            reward += self.move_cost  # Add move cost
        # else: STAY keeps floor and mask unchanged

        next_state = (floor, new_mask)
        return {next_state: (1.0, reward)}

# ============================================================
# VALUE ITERATION SOLVER
# ============================================================

class ValueIteration:
    """Solves MDP using Value Iteration algorithm"""

    def __init__(self, mdp, theta=1e-6):
        self.mdp = mdp
        self.theta = theta

    def solve(self, verbose=False):
        """Solve the MDP and return value function and policy"""
        states = self.mdp.get_states()
        V = defaultdict(float)
        policy = {}

        iteration = 0
        while True:
            delta = 0
            iteration += 1

            for state in states:
                old_value = V[state]

                # Compute max value over actions
                max_value = float('-inf')
                for action in self.mdp.actions:
                    if self.mdp.is_valid_action(state, action):
                        action_value = self._compute_action_value(V, state, action)
                        max_value = max(max_value, action_value)

                V[state] = max_value if max_value != float('-inf') else 0
                delta = max(delta, abs(old_value - V[state]))

            if verbose and iteration % 5 == 0:
                print(f"  Iteration {iteration}, Delta: {delta:.6f}")

            if delta < self.theta:
                break

        # Extract policy
        for state in states:
            best_action = None
            best_value = float('-inf')
            for action in self.mdp.actions:
                if self.mdp.is_valid_action(state, action):
                    action_value = self._compute_action_value(V, state, action)
                    if action_value > best_value:
                        best_value = action_value
                        best_action = action
            if best_action:
                policy[state] = best_action

        if verbose:
            print(f"  ✅ Converged in {iteration} iterations")

        return dict(V), policy

    def _compute_action_value(self, V, state, action):
        """Compute Q-value for state-action pair"""
        transitions = self.mdp.get_transitions(state, action)
        value = 0
        for next_state, (prob, reward) in transitions.items():
            value += prob * (reward + self.mdp.discount_factor * V[next_state])
        return value


# ============================================================
# TEXT VISUALIZER
# ============================================================

class EnhancedTextVisualizer:
    """Text-based visualization of elevator simulation"""

    def __init__(self, mdp, policy, V):
        self.mdp = mdp
        self.policy = policy
        self.V = V

    def simulate(self, initial_state, max_steps=40, delay=0.5):
        """Run text-based simulation"""
        state = initial_state
        total_reward = 0

        print("\n" + "="*80)
        print(" ELEVATOR SIMULATION (Text Mode)")
        print("="*80)
        print(f"Initial State: Floor {initial_state[0]}, Requests: {bin(initial_state[1])[2:].zfill(3)}\n")

        for step in range(max_steps):
            floor, mask = state
            action = self.policy.get(state, 'STAY')

            # Validate action
            if not self.mdp.is_valid_action(state, action):
                action = 'STAY'

            next_state, reward = self.mdp.step(state, action, deterministic=False)

            # Update state and accumulate reward
            total_reward += reward

            # Print step details
            next_floor, next_mask = next_state
            request_count = self.mdp.count_requests(mask)
            next_request_count = self.mdp.count_requests(next_mask)

            status = "✓" if reward >= 0 else "✗"

            print(f"Step {step+1:2d} {status} | Floor: {floor} → {next_floor} | "
                  f"Action: {action:6s} | Reward: {reward:+.1f} | "
                  f"Total: {total_reward:+.1f} | Requests: {request_count} → {next_request_count}")

            state = next_state
            time.sleep(delay)

        print("\n" + "="*80)
        print(f"SIMULATION COMPLETE")
        print(f"Total Steps: {max_steps}")
        print(f"Final Reward: {total_reward:.2f}")
        print(f"Average Reward per Step: {total_reward/max_steps:.2f}")
        print("="*80 + "\n")


# ============================================================
# POLICY ANALYSIS
# ============================================================

def print_policy_analysis(mdp, policy, V):
    """Print analysis of learned policy"""
    print("\n" + "="*60)
    print("POLICY ANALYSIS")
    print("="*60)

    print("\nOptimal Actions by Floor (no requests):")
    for floor in range(1, mdp.num_floors + 1):
        state = (floor, 0)
        action = policy.get(state, '-')
        value = V.get(state, 0)
        print(f"  Floor {floor}: {action:6s} (V={value:6.2f})")

    print(f"\nOptimal Actions by Floor (all requests):")
    for floor in range(1, mdp.num_floors + 1):
        mask = (2 ** mdp.num_floors) - 1  # All bits set
        state = (floor, mask)
        action = policy.get(state, '-')
        value = V.get(state, 0)
        print(f"  Floor {floor}: {action:6s} (V={value:6.2f})")

    print(f"\nMDP Parameters:")
    print(f"  Discount Factor: {mdp.discount_factor}")
    print(f"  Move Cost: {mdp.move_cost}")
    print(f"  Request Penalty: {mdp.request_penalty}")
    print(f"  Pick Reward: {mdp.pick_reward}")
    print(f"  New Request Probability: {mdp.new_request_prob}")
    print(f"\nTotal States: {len(policy)}")
    print("="*60)


# ============================================================
# GUI
# ============================================================

class ElevatorGUI:
    """Graphical interface for Elevator MDP visualization"""

    def __init__(self, mdp, policy, V):
        self.mdp = mdp
        self.policy = policy
        self.V = V

        # Simulation state
        self.current_state = (1, 0)
        self.total_reward = 0
        self.step_count = 0
        self.running = False
        self.speed = 500  # milliseconds between steps

        # Setup GUI
        self.setup_gui()

    def setup_gui(self):
        """Create the main GUI window"""
        self.root = tk.Tk()
        self.root.title("🏢 Elevator MDP Simulation")
        self.root.geometry("900x750")
        self.root.configure(bg='#1a1a2e')

        # Main container
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        title = tk.Label(
            main_frame,
            text="🏢 ELEVATOR MDP SIMULATOR 🏢",
            font=('Arial', 20, 'bold'),
            bg='#1a1a2e',
            fg='#00d4ff'
        )
        title.pack(pady=(0, 20))

        # Content frame (building + stats side by side)
        content_frame = tk.Frame(main_frame, bg='#1a1a2e')
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Building canvas (left side)
        self.create_building_canvas(content_frame)

        # Stats panel (right side)
        self.create_stats_panel(content_frame)

        # Control panel (bottom)
        self.create_control_panel(main_frame)

        # Initialize display
        self.update_display()

    def create_building_canvas(self, parent):
        """Create the building visualization canvas"""
        building_frame = tk.Frame(parent, bg='#16213e', relief=tk.RIDGE, bd=3)
        building_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Canvas for drawing
        self.canvas = tk.Canvas(
            building_frame,
            width=350,
            height=450,
            bg='#16213e',
            highlightthickness=0
        )
        self.canvas.pack(pady=20, padx=20)

        # Draw static building structure
        self.draw_building_structure()

    def draw_building_structure(self):
        """Draw the static building elements"""
        self.canvas.delete("all")

        # Building outline
        self.canvas.create_rectangle(50, 30, 300, 420, outline='#4a5568', width=3)

        # Floor labels and dividers
        floor_height = 120
        for i, floor in enumerate([3, 2, 1]):
            y = 50 + i * floor_height

            # Floor divider
            if i > 0:
                self.canvas.create_line(50, y, 300, y, fill='#4a5568', width=2)

            # Floor label
            self.canvas.create_text(
                30, y + floor_height//2,
                text=f"F{floor}",
                font=('Arial', 14, 'bold'),
                fill='#a0aec0'
            )

        # Elevator shaft
        self.canvas.create_rectangle(120, 40, 220, 410, outline='#718096', width=2)

        # Shaft label
        self.canvas.create_text(
            170, 20,
            text="SHAFT",
            font=('Arial', 10),
            fill='#718096'
        )

    def draw_elevator(self, floor):
        """Draw the elevator at specified floor"""
        # Clear previous elevator
        self.canvas.delete("elevator")

        # Calculate Y position
        floor_height = 120
        floor_positions = {3: 50, 2: 170, 1: 290}
        y = floor_positions[floor]

        # Draw elevator car
        self.canvas.create_rectangle(
            130, y + 10,
            210, y + 100,
            fill='#48bb78',
            outline='#2f855a',
            width=3,
            tags="elevator"
        )

        # Elevator icon
        self.canvas.create_text(
            170, y + 55,
            text="🛗",
            font=('Arial', 30),
            tags="elevator"
        )

        # Floor indicator on elevator
        self.canvas.create_text(
            170, y + 90,
            text=f"F{floor}",
            font=('Arial', 10, 'bold'),
            fill='white',
            tags="elevator"
        )

    def draw_requests(self, mask):
        """Draw waiting passengers"""
        self.canvas.delete("requests")

        floor_height = 120
        floor_positions = {3: 50, 2: 170, 1: 290}

        for floor in [1, 2, 3]:
            y = floor_positions[floor]

            if self.mdp.has_request(mask, floor):
                # Waiting person
                self.canvas.create_text(
                    260, y + 55,
                    text="🧍",
                    font=('Arial', 25),
                    tags="requests"
                )
                # Request indicator
                self.canvas.create_oval(
                    270, y + 20, 290, y + 40,
                    fill='#fc8181',
                    outline='#c53030',
                    width=2,
                    tags="requests"
                )
                self.canvas.create_text(
                    280, y + 30,
                    text="!",
                    font=('Arial', 12, 'bold'),
                    fill='white',
                    tags="requests"
                )
            else:
                # Empty indicator
                self.canvas.create_oval(
                    270, y + 20, 290, y + 40,
                    fill='#4a5568',
                    outline='#2d3748',
                    width=2,
                    tags="requests"
                )

    def create_stats_panel(self, parent):
        """Create the statistics panel"""
        stats_frame = tk.Frame(parent, bg='#16213e', relief=tk.RIDGE, bd=3)
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Title
        tk.Label(
            stats_frame,
            text="📊 STATISTICS",
            font=('Arial', 14, 'bold'),
            bg='#16213e',
            fg='#00d4ff'
        ).pack(pady=(20, 15))

        # Stats container
        stats_container = tk.Frame(stats_frame, bg='#16213e')
        stats_container.pack(fill=tk.X, padx=15)

        # Create stat labels
        self.stat_labels = {}
        stats_config = [
            ('step', '🔢 Step:', '0'),
            ('floor', '📍 Elevator:', 'Floor 1'),
            ('requests', '📋 Requests:', '0'),
            ('reward', '💰 Total Reward:', '0.00'),
            ('last_reward', '📈 Last Reward:', '+0.00'),
            ('action', '🎬 Action:', '-'),
            ('optimal', '🎯 Optimal:', '-'),
            ('value', '📊 State Value:', '0.00'),
        ]

        for key, label_text, default in stats_config:
            frame = tk.Frame(stats_container, bg='#16213e')
            frame.pack(fill=tk.X, pady=5)

            tk.Label(
                frame,
                text=label_text,
                font=('Arial', 11),
                bg='#16213e',
                fg='#a0aec0',
                anchor='w',
                width=15
            ).pack(side=tk.LEFT)

            value_label = tk.Label(
                frame,
                text=default,
                font=('Arial', 11, 'bold'),
                bg='#16213e',
                fg='#48bb78',
                anchor='e',
                width=10
            )
            value_label.pack(side=tk.RIGHT)
            self.stat_labels[key] = value_label

        # Request details
        tk.Label(
            stats_frame,
            text="─── Floor Status ───",
            font=('Arial', 10),
            bg='#16213e',
            fg='#718096'
        ).pack(pady=(20, 10))

        self.floor_status = {}
        for floor in [3, 2, 1]:
            frame = tk.Frame(stats_frame, bg='#16213e')
            frame.pack(fill=tk.X, padx=20)

            tk.Label(
                frame,
                text=f"Floor {floor}:",
                font=('Arial', 10),
                bg='#16213e',
                fg='#a0aec0'
            ).pack(side=tk.LEFT)

            status = tk.Label(
                frame,
                text="⚪ Empty",
                font=('Arial', 10),
                bg='#16213e',
                fg='#48bb78'
            )
            status.pack(side=tk.RIGHT)
            self.floor_status[floor] = status

    def create_control_panel(self, parent):
        """Create the control panel"""
        control_frame = tk.Frame(parent, bg='#1a1a2e')
        control_frame.pack(fill=tk.X, pady=(20, 0))

        # Button style
        btn_style = {
            'font': ('Arial', 11, 'bold'),
            'width': 12,
            'height': 2,
            'relief': tk.RAISED,
            'bd': 2
        }

        # Buttons frame
        btn_frame = tk.Frame(control_frame, bg='#1a1a2e')
        btn_frame.pack()

        # Start/Stop button
        self.start_btn = tk.Button(
            btn_frame,
            text="▶ START",
            bg='#48bb78',
            fg='white',
            command=self.toggle_simulation,
            **btn_style
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        # Step button
        self.step_btn = tk.Button(
            btn_frame,
            text="⏭ STEP",
            bg='#4299e1',
            fg='white',
            command=self.single_step,
            **btn_style
        )
        self.step_btn.pack(side=tk.LEFT, padx=5)

        # Reset button
        tk.Button(
            btn_frame,
            text="🔄 RESET",
            bg='#ed8936',
            fg='white',
            command=self.reset_simulation,
            **btn_style
        ).pack(side=tk.LEFT, padx=5)

        # Random state button
        tk.Button(
            btn_frame,
            text="🎲 RANDOM",
            bg='#9f7aea',
            fg='white',
            command=self.random_state,
            **btn_style
        ).pack(side=tk.LEFT, padx=5)

        # Speed control
        speed_frame = tk.Frame(control_frame, bg='#1a1a2e')
        speed_frame.pack(pady=(15, 0))

        tk.Label(
            speed_frame,
            text="Speed:",
            font=('Arial', 10),
            bg='#1a1a2e',
            fg='#a0aec0'
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.speed_scale = tk.Scale(
            speed_frame,
            from_=100,
            to=2000,
            orient=tk.HORIZONTAL,
            length=200,
            bg='#16213e',
            fg='#a0aec0',
            highlightthickness=0,
            command=self.update_speed
        )
        self.speed_scale.set(500)
        self.speed_scale.pack(side=tk.LEFT)

        tk.Label(
            speed_frame,
            text="ms",
            font=('Arial', 10),
            bg='#1a1a2e',
            fg='#a0aec0'
        ).pack(side=tk.LEFT, padx=(5, 0))

    def update_display(self):
        """Update all visual elements"""
        floor, mask = self.current_state

        # Update building visualization
        self.draw_building_structure()
        self.draw_elevator(floor)
        self.draw_requests(mask)

        # Update stats
        self.stat_labels['step'].config(text=str(self.step_count))
        self.stat_labels['floor'].config(text=f"Floor {floor}")
        self.stat_labels['requests'].config(
            text=str(self.mdp.count_requests(mask)),
            fg='#fc8181' if self.mdp.count_requests(mask) > 0 else '#48bb78'
        )
        self.stat_labels['reward'].config(text=f"{self.total_reward:.2f}")
        self.stat_labels['optimal'].config(text=self.policy.get(self.current_state, '-'))
        self.stat_labels['value'].config(text=f"{self.V.get(self.current_state, 0):.2f}")

        # Update floor status
        for f in [1, 2, 3]:
            if self.mdp.has_request(mask, f):
                self.floor_status[f].config(text="🔴 Waiting", fg='#fc8181')
            else:
                self.floor_status[f].config(text="⚪ Empty", fg='#48bb78')

    def single_step(self):
        """Execute a single step"""
        # Get action from policy with safe default
        action = self.policy.get(self.current_state, 'STAY')

        # Validate action
        if not self.mdp.is_valid_action(self.current_state, action):
            action = 'STAY'

        next_state, reward = self.mdp.step(self.current_state, action, deterministic=False)

        # Update state
        self.current_state = next_state
        self.total_reward += reward
        self.step_count += 1

        # Update action display
        action_colors = {
            'UP': '#48bb78',
            'DOWN': '#4299e1',
            'STAY': '#ed8936',
            'PICK': '#9f7aea'
        }
        self.stat_labels['action'].config(
            text=action,
            fg=action_colors.get(action, '#48bb78')
        )
        self.stat_labels['last_reward'].config(
            text=f"{reward:+.2f}",
            fg='#48bb78' if reward >= 0 else '#fc8181'
        )

        self.update_display()

    def toggle_simulation(self):
        """Start or stop continuous simulation"""
        self.running = not self.running

        if self.running:
            self.start_btn.config(text="⏸ PAUSE", bg='#fc8181')
            self.step_btn.config(state=tk.DISABLED)
            self.run_simulation()
        else:
            self.start_btn.config(text="▶ START", bg='#48bb78')
            self.step_btn.config(state=tk.NORMAL)

    def run_simulation(self):
        """Run continuous simulation"""
        if self.running:
            self.single_step()
            self.root.after(self.speed, self.run_simulation)

    def reset_simulation(self):
        """Reset to initial state"""
        self.running = False
        self.start_btn.config(text="▶ START", bg='#48bb78')
        self.step_btn.config(state=tk.NORMAL)

        self.current_state = (1, 0)
        self.total_reward = 0
        self.step_count = 0

        self.stat_labels['action'].config(text="-")
        self.stat_labels['last_reward'].config(text="+0.00", fg='#48bb78')

        self.update_display()

    def random_state(self):
        """Set a random state from valid policy states"""
        valid_states = list(self.policy.keys())
        if valid_states:
            self.current_state = random.choice(valid_states)
            self.total_reward = 0
            self.step_count = 0
            self.stat_labels['last_reward'].config(text="+0.00", fg='#48bb78')
            self.update_display()

    def update_speed(self, value):
        """Update simulation speed"""
        self.speed = int(value)

    def run(self):
        """Start the GUI"""
        self.root.mainloop()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    """Main function with GUI option"""
    print("="*60)
    print("   ELEVATOR MDP PROJECT")
    print("   3-Floor Building, Single Elevator")
    print("   NEW REWARD STRUCTURE:")
    print("   • -1 per timestep for each pending request")
    print("   • -0.5 for moving (energy cost)")
    print("   • +5 for picking up passengers")
    print("="*60)

    # Initialize MDP with new reward structure
    mdp = ElevatorMDP(
        num_floors=3,
        new_request_prob=0.1,
        discount_factor=0.95,
        move_cost=-0.5,
        request_penalty=-1.0,  # -1 per pending request
        pick_reward=5.0
    )

    # Solve using Value Iteration
    print("\nSolving MDP...")
    vi_solver = ValueIteration(mdp, theta=1e-8)
    V, policy = vi_solver.solve(verbose=True)

    # Print policy summary
    print_policy_analysis(mdp, policy, V)

    # Choose visualization mode
    print("\n" + "="*60)
    print("VISUALIZATION OPTIONS")
    print("="*60)
    print("\n1. 🖥️  Graphical GUI (Tkinter)")
    print("2. 📟 Enhanced Text Animation")
    print("3. ⏭️  Skip Visualization")

    choice = input("\nSelect option (1/2/3): ").strip()

    if choice == '1':
        print("\n✨ Launching GUI...")
        gui = ElevatorGUI(mdp, policy, V)
        gui.run()

    elif choice == '2':
        print("\n▶️ Starting text simulation...")
        visualizer = EnhancedTextVisualizer(mdp, policy, V)
        # Test with a state that has some requests
        visualizer.simulate(initial_state=(1, 0b101), max_steps=20, delay=0.5)

    else:
        print("\n⏭️ Visualization skipped.")

    print("\n✅ Program complete!")


if __name__ == "__main__":
    main()
