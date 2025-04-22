import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.gaussian_process.kernels import RBF

# 1. Generate synthetic test scenarios (same as before)
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
        imbalance = bid_size / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0.5
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

# 3. Use the GPIRL model to predict actions
def predict_actions_gpirl(test_features, train_features, alpha, kernel_param=1.0):
    """
    Use the learned GP weights to predict actions
    
    Parameters:
    test_features - features of test states
    train_features - features of training states used for GP
    alpha - GP weights from GPIRL
    kernel_param - RBF kernel parameter
    
    Returns:
    predictions - array of predicted actions
    """
    # Create kernel function
    kernel = RBF(length_scale=kernel_param)
    
    # Compute kernel matrix between test and training points
    K_test = kernel(test_features, train_features)
    
    # Compute rewards
    rewards = K_test @ alpha
    
    # Predict actions based on rewards and features
    predictions = []
    
    for i, reward in enumerate(rewards):
        # Get state features for this example
        features = test_features[i]
        spread = features[4]  # Spread is the 5th feature
        imbalance = features[5]  # Imbalance is the 6th feature
        
        # Simple policy - can be refined based on your domain knowledge
        if reward > 0.2 and imbalance > 0.6:
            predictions.append('BUY')
        elif reward > 0.2 and imbalance < 0.4:
            predictions.append('SELL')
        else:
            predictions.append('WAIT')
    
    return predictions, rewards

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

# 5. Visualize GPIRL results
def visualize_gpirl_results(test_scenarios, true_actions, predicted_actions, rewards):
    """
    Create visualizations to interpret GPIRL model performance
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
    
    # Plot 2: Reward Distribution
    axes[0, 1].hist(rewards, bins=30)
    axes[0, 1].set_title('GPIRL Reward Distribution')
    axes[0, 1].set_xlabel('Reward')
    axes[0, 1].set_ylabel('Frequency')
    
    # Plot 3: True actions by imbalance and spread
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
    
    # Plot 4: Predicted actions by imbalance and spread
    for action in ['BUY', 'SELL', 'WAIT']:
        mask = np.array(predicted_actions) == action
        axes[1, 1].scatter(imbalances[mask], spreads[mask], 
                         c=colors[action], label=f'Predicted {action}', alpha=0.5)
    
    axes[1, 1].set_xlabel('Order Book Imbalance')
    axes[1, 1].set_ylabel('Bid-Ask Spread')
    axes[1, 1].set_title('Predicted Actions by Imbalance and Spread')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('gpirl_trading_model_evaluation.png')
    plt.show()
    
    # Additional plot: Rewards mapped to state space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(imbalances, spreads, c=rewards, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Reward')
    plt.xlabel('Order Book Imbalance')
    plt.ylabel('Bid-Ask Spread')
    plt.title('GPIRL Rewards Mapped to State Space')
    plt.grid(True, alpha=0.3)
    plt.savefig('gpirl_reward_landscape.png')
    plt.show()

# 6. Backtest trading strategy
def backtest_gpirl_strategy(test_scenarios, predicted_actions, starting_capital=10000):
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
    plt.title(f'GPIRL Strategy Equity Curve\nReturn: {total_return:.2%}, Trades: {n_trades}')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.savefig('gpirl_equity_curve.png')
    plt.show()
    
    return {
        'final_capital': capital,
        'total_return': total_return,
        'n_trades': n_trades,
        'trades': trades,
        'equity_curve': equity_curve
    }

# 7. Main test function for GPIRL
def test_gpirl_trading_model(gpirl_alpha, train_features_scaled, scaler):
    """
    Test GPIRL trading model with synthetic scenarios
    
    Parameters:
    gpirl_alpha - GP weights from training
    train_features_scaled - scaled features used during training
    scaler - feature scaler
    """
    # Generate test scenarios
    test_scenarios, true_actions = generate_test_scenarios(n_samples=1000)
    
    # Convert to feature representation
    test_features = np.vstack([state_to_features(state) for state in test_scenarios])
    test_features_scaled = scaler.transform(test_features)
    
    # Predict actions using GPIRL model
    predicted_actions, rewards = predict_actions_gpirl(
        test_features_scaled, 
        train_features_scaled, 
        gpirl_alpha
    )
    
    # Evaluate model performance
    accuracy, cm, report = evaluate_model(true_actions, predicted_actions)
    print(f"GPIRL Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Visualize results
    visualize_gpirl_results(test_scenarios, true_actions, predicted_actions, rewards)
    
    # Backtest trading strategy
    backtest_results = backtest_gpirl_strategy(test_scenarios, predicted_actions)
    
    print(f"Trading backtest return: {backtest_results['total_return']:.2%}")
    print(f"Number of trades: {backtest_results['n_trades']}")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'report': report,
        'test_scenarios': test_scenarios,
        'true_actions': true_actions,
        'predicted_actions': predicted_actions,
        'rewards': rewards,
        'backtest_results': backtest_results
    }

# 8. Compare GPIRL with LNIRL
def compare_irl_models(lnirl_weights, gpirl_alpha, X_train_scaled, scaler):
    """
    Compare performance of LNIRL and GPIRL models
    """
    # Generate test data
    test_scenarios, true_actions = generate_test_scenarios(n_samples=1000)
    test_features = np.vstack([state_to_features(state) for state in test_scenarios])
    test_features_scaled = scaler.transform(test_features)
    
    # LNIRL predictions (using function from previous artifact)
    def predict_lnirl_actions(states, weights, scaler):
        features = np.vstack([state_to_features(state) for state in states])
        scaled_features = scaler.transform(features)
        rewards = np.dot(scaled_features, weights)
        
        predictions = []
        for i, reward in enumerate(rewards):
            bid_price, bid_size, ask_price, ask_size = states[i]
            spread = ask_price - bid_price
            imbalance = bid_size / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0.5
            
            if reward > 0.5 and imbalance > 0.6:
                predictions.append('BUY')
            elif reward > 0.5 and imbalance < 0.4:
                predictions.append('SELL')
            else:
                predictions.append('WAIT')
        
        return predictions
    
    # GPIRL predictions
    gpirl_predictions, gpirl_rewards = predict_actions_gpirl(
        test_features_scaled, 
        X_train_scaled, 
        gpirl_alpha
    )
    
    # LNIRL predictions
    lnirl_predictions = predict_lnirl_actions(test_scenarios, lnirl_weights, scaler)
    
    # Evaluate both models
    gpirl_accuracy, _, gpirl_report = evaluate_model(true_actions, gpirl_predictions)
    lnirl_accuracy, _, lnirl_report = evaluate_model(true_actions, lnirl_predictions)
    
    # Print comparison
    print("\n===== MODEL COMPARISON =====")
    print(f"GPIRL Accuracy: {gpirl_accuracy:.4f}")
    print(f"LNIRL Accuracy: {lnirl_accuracy:.4f}")
    
    # Calculate agreement between models
    agreement = sum(1 for gp, ln in zip(gpirl_predictions, lnirl_predictions) if gp == ln) / len(gpirl_predictions)
    print(f"Model Agreement: {agreement:.4f}")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Confusion matrices
    cm_gpirl = confusion_matrix([action_map[a] for a in true_actions], 
                               [action_map[a] for a in gpirl_predictions])
    cm_lnirl = confusion_matrix([action_map[a] for a in true_actions], 
                               [action_map[a] for a in lnirl_predictions])
    
    sns.heatmap(cm_gpirl, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['WAIT', 'BUY', 'SELL'],
                yticklabels=['WAIT', 'BUY', 'SELL'],
                ax=axes[0])
    axes[0].set_title('GPIRL Confusion Matrix')
    
    sns.heatmap(cm_lnirl, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['WAIT', 'BUY', 'SELL'],
                yticklabels=['WAIT', 'BUY', 'SELL'],
                ax=axes[1])
    axes[1].set_title('LNIRL Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('irl_model_comparison.png')
    plt.show()
    
    return {
        'gpirl_accuracy': gpirl_accuracy,
        'lnirl_accuracy': lnirl_accuracy,
        'agreement': agreement,
        'gpirl_predictions': gpirl_predictions,
        'lnirl_predictions': lnirl_predictions
    }

# Example usage
if __name__ == "__main__":
    # Replace with your actual trained weights
    example_gpirl_alpha = np.random.rand(100)  # Mock GP weights, size would match your training data
    example_lnirl_weights = np.array([0.1, 0.3, -0.1, 0.2, -0.4, 0.8])  # Mock LNIRL weights
    
    # Create mock training data and scaler for testing
    from sklearn.preprocessing import StandardScaler
    mock_train_features = np.random.rand(100, 6)
    mock_scaler = StandardScaler()
    mock_train_scaled = mock_scaler.fit_transform(mock_train_features)
    
    # Define global action map
    action_map = {'WAIT': 0, 'BUY': 1, 'SELL': 2}
    
    # Run GPIRL tests
    gpirl_results = test_gpirl_trading_model(
        example_gpirl_alpha, 
        mock_train_scaled, 
        mock_scaler
    )
    
    # Compare models (uncomment when both models are available)
    # comparison = compare_irl_models(
    #     example_lnirl_weights,
    #     example_gpirl_alpha,
    #     mock_train_scaled,
    #     mock_scaler
    # )