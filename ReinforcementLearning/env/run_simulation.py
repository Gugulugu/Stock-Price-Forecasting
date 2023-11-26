from stock_trading_env import StockTradingEnv

PATH = '/home/dz/Stocks/ReinforcementLearning/data/test/STOCKS_GOOGL.csv'
# Create an instance of the environment
env = StockTradingEnv(csv_file=PATH)

# Run for a certain number of episodes
num_episodes = 10
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Sample a random action (for demonstration)
        action = env.action_space.sample()

        # Execute the action
        next_state, reward, done, _ = env.step(action)

        # Optionally render the environment
        #env.render()

        # Update state
        state = next_state
    
    # Print episode results
    print('Episode %d finished after %d timesteps' % (episode+1, env.current_step))
    print('Final capital: %.2f\n' % env.capital)

    # Plot the final state of the first epoch
    if episode == 0:
        env.plot_final_chart()


# Close the environment
env.close()
