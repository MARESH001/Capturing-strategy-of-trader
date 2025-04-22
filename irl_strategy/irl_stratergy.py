import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 1. Generate synthetic test data with known strategies
def generate_test_scenarios(n_samples=1000):
    """
    Generate synthetic order book scenarios with known optimal strategies
    """
    np.random.seed(42)
    test_scenarios = []
    true_actions = []
    
    for _ in range(n_samples):
        # Generate random order book state
        bid_price = np.random.uniform(90, 100)
        ask_price = bid_price + np.random.uniform(0.01, 1.0)  # Ensure ask > bid
        bid_size = np.random.randint(1, 100)
        ask_size = np.random.randint(1, 100)
        
        # Create state tuple
        state = (bid_price, bid_size, ask_price, ask_size)
        
        # Apply strategic rules (these represent "true" optimal strategies):
        # Rule 1: Large bid/ask imbalance suggests directional pressure
        imbalance = bid_size / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0.5
        
        # Rule 2: Tight spread may indicate good liquidity
        spread = ask_price - bid_price
        
        # Apply strategy rules
        if imbalance > 0.7 and spread < 0.2:
            # Strong buying pressure with tight spread -> BUY
            action = 'BUY'
        elif imbalance < 0.3 and spread < 0.2:
            # Strong selling pressure with tight spread -> SELL
            action = 'SELL'
        elif spread > 0.5:
            # Wide spread -> WAIT for better entry
            action = 'WAIT'
        else:
            # Balanced book -> WAIT
            action = 'WAIT'
        
        test_scenarios.append(state)
        true_actions.append(action)
    
    return test_scenarios, true_actions

# 2. Function to convert state to features (same as in your code)
def state_to_features(state):
    # Extract bid-ask spread, sizes, and other features
    bid_price, bid_size, ask_price, ask_size = state
    spread = ask_price - bid_price
    imbalance = bid_size / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0.5
    
    return np.array([bid_price, bid_size, ask_price, ask_size, spread, imbalance])

# 3. Use the trained model to predict actions
def predict_actions(states, weights, scaler):
    """
    Use the learned reward weights to predict actions
    """
    # Convert states to feature vectors
    features = np.vstack([state_to_features(state) for state in states])
    
    # Scale features using the same scaler used during training
    scaled_features = scaler.transform(features)
    
    # Calculate rewards for each state
    rewards = np.dot(scaled_features, weights)
    
    # Simple policy: Choose action with highest expected reward
    # In a real system, you'd simulate different actions and choose the one with highest reward
    # For this test, we'll create a simple heuristic based on the reward value
    predictions = []
    
    for i, reward in enumerate(rewards):
        # Get state features for this example
        bid_price, bid_size, ask_price, ask_size = states[i]
        spread = ask_price - bid_price
        imbalance = bid_size / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0.5
        
        # Use reward and features to decide action
        if reward > 0.5 and imbalance > 0.6:
            predictions.append('BUY')
        elif reward > 0.5 and imbalance < 0.4:
            predictions.append('SELL')
        else:
            predictions.append('WAIT')
    
    return predictions

# 4. Evaluate model performance
def evaluate_model(true_actions, predicted_actions):
    """
    Evaluate model performance against true actions
    """
    # Calculate accuracy
    accuracy = sum(1 for t, p in zip(true_actions, predicted_actions) if t == p) / len(true_actions)
    
    # Create confusion matrix
    action_map = {'WAIT': 0, 'BUY': 1, 'SELL': 2}
    true_encoded = [action_map[a] for a in true_actions]
    pred_encoded = [action_map[a] for a in predicted_actions]
    
    cm = confusion_matrix(true_encoded, pred_encoded)
    
    # Classification report
    report = classification_report(true_actions, predicted_actions)
    
    return accuracy, cm, report

# 5. Visualize results
def visualize_results(test_scenarios, true_actions, predicted_actions, weights):
    """
    Create visualizations to interpret model performance
    """
    features = np.vstack([state_to_features(state) for state in test_scenarios])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Confusion Matrix
    action_map = {'WAIT': 0, 'BUY': 1, 'SELL': 2}
    true_encoded = [action_map[a] for a in true_actions]
    pred_encoded = [action_map[a] for a in predicted_actions]
    
    cm = confusion_matrix(true_encoded, pred_encoded)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['WAIT', 'BUY', 'SELL'],
                yticklabels=['WAIT', 'BUY', 'SELL'],
                ax=axes[0, 0])
    axes[0, 0].set_ylabel('True Actions')
    axes[0, 0].set_xlabel('Predicted Actions')
    axes[0, 0].set_title('Confusion Matrix')
    
    # Plot 2: Feature Importance (weights)
    feature_names = ['Bid Price', 'Bid Size', 'Ask Price', 'Ask Size', 'Spread', 'Imbalance']
    axes[0, 1].bar(feature_names, weights)
    axes[0, 1].set_title('Feature Importance (Reward Weights)')
    axes[0, 1].set_xticklabels(feature_names, rotation=45)
    
    # Plot 3: Action distribution by imbalance and spread
    colors = {'BUY': 'green', 'SELL': 'red', 'WAIT': 'blue'}
    spreads = features[:, 4]
    imbalances = features[:, 5]
    
    for action in ['BUY', 'SELL', 'WAIT']:
        mask = np.array(true_actions) == action
        axes[1, 0].scatter(imbalances[mask], spreads[mask], 
                         c=colors[action], label=f'True {action}', alpha=0.5)
    
    axes[1, 0].set_xlabel('Order Book Imbalance')
    axes[1, 0].set_ylabel('Bid-Ask Spread')
    axes[1, 0].set_title('True Actions by Imbalance and Spread')
    axes[1, 0].legend()
    
    # Plot 4: Predicted vs True Actions
    for action in ['BUY', 'SELL', 'WAIT']:
        mask = np.array(predicted_actions) == action
        axes[1, 1].scatter(imbalances[mask], spreads[mask], 
                         c=colors[action], label=f'Predicted {action}', alpha=0.5)
    
    axes[1, 1].set_xlabel('Order Book Imbalance')
    axes[1, 1].set_ylabel('Bid-Ask Spread')
    axes[1, 1].set_title('Predicted Actions by Imbalance and Spread')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('irl_trading_model_evaluation.png')
    plt.show()

# Main test function
def test_irl_trading_model(weights, scaler):
    # 1. Generate test scenarios
    test_scenarios, true_actions = generate_test_scenarios(n_samples=1000)
    
    # 2. Predict actions using learned weights
    predicted_actions = predict_actions(test_scenarios, weights, scaler)
    
    # 3. Evaluate model performance
    accuracy, cm, report = evaluate_model(true_actions, predicted_actions)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # 4. Visualize results
    visualize_results(test_scenarios, true_actions, predicted_actions, weights)
    
    # 5. Return performance metrics for further analysis
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'report': report,
        'test_scenarios': test_scenarios,
        'true_actions': true_actions,
        'predicted_actions': predicted_actions
    }

# Bonus: Backtest trading performance
def backtest_trading_strategy(test_scenarios, predicted_actions, starting_capital=10000):
    """
    Simulate trading based on predicted actions and measure performance
    """
    capital = starting_capital
    position = 0  # 0 = no position, positive = long, negative = short
    trades = []
    equity_curve = [capital]
    
    for i, (state, action) in enumerate(zip(test_scenarios, predicted_actions)):
        bid_price, bid_size, ask_price, ask_size = state
        
        # Simple execution model:
        # - Buy at ask price
        # - Sell at bid price
        # - Position size is fixed at 1 unit
        
        if action == 'BUY' and position <= 0:
            # Close any short position first
            if position < 0:
                capital += position * bid_price  # Cover short at bid
                trades.append(('COVER', i, bid_price))
                
            # Enter long position
            position = 1
            capital -= ask_price  # Pay ask price to buy
            trades.append(('BUY', i, ask_price))
            
        elif action == 'SELL' and position >= 0:
            # Close any long position first
            if position > 0:
                capital += position * bid_price  # Sell at bid
                trades.append(('SELL', i, bid_price))
                
            # Enter short position
            position = -1
            capital += ask_price  # Receive ask price when shorting
            trades.append(('SHORT', i, ask_price))
        
        # Update equity (mark-to-market)
        mid_price = (bid_price + ask_price) / 2
        current_equity = capital + position * mid_price
        equity_curve.append(current_equity)
    
    # Close final position at last price
    if position != 0:
        final_state = test_scenarios[-1]
        final_bid, _, final_ask, _ = final_state
        
        if position > 0:
            capital += position * final_bid  # Sell at bid
        else:
            capital -= position * final_ask  # Cover at ask
    
    # Calculate performance metrics
    total_return = (capital - starting_capital) / starting_capital
    n_trades = len(trades)
    
    # Plot equity curve
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve)
    plt.title(f'Strategy Equity Curve\nReturn: {total_return:.2%}, Trades: {n_trades}')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.savefig('equity_curve.png')
    plt.show()
    
    return {
        'final_capital': capital,
        'total_return': total_return,
        'n_trades': n_trades,
        'trades': trades,
        'equity_curve': equity_curve
    }

# Run the complete test and evaluation
# In your actual code, you would call:
# results = test_irl_trading_model(lnirl_weights, scaler)
# backtest_results = backtest_trading_strategy(results['test_scenarios'], results['predicted_actions'])

# Example usage (assuming your model is already trained):
if __name__ == "__main__":
    # Replace with your actual trained weights and scaler
    example_weights = np.array([0.1, 0.3, -0.1, 0.2, -0.4, 0.8])  # These should be your actual LNIRL weights
    
    # Create a mock scaler if needed for testing
    from sklearn.preprocessing import StandardScaler
    mock_scaler = StandardScaler()
    mock_X = np.random.rand(100, 6)
    mock_scaler.fit(mock_X)
    
    # Run tests
    test_results = test_irl_trading_model(example_weights, mock_scaler)
    backtest_results = backtest_trading_strategy(
        test_results['test_scenarios'], 
        test_results['predicted_actions']
    )
    
    print(f"Trading backtest return: {backtest_results['total_return']:.2%}")
    print(f"Number of trades: {backtest_results['n_trades']}")