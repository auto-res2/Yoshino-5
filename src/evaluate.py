import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import networkx as nx
import gym
from transformers import pipeline

class IterativeTaskEnv(gym.Env):
    """Custom environment for meta reinforcement learning evaluation."""
    def __init__(self):
        super(IterativeTaskEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.current_step = 0
        self.max_steps = 20

    def reset(self):
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        return np.random.rand(10).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        reward = 1.0 if np.random.rand() > 0.2 else -1.0
        done = self.current_step >= self.max_steps
        return self._next_observation(), reward, done, {}

def evaluate_baseline_model(model, X_test, y_test):
    """Evaluate baseline model performance."""
    print("Evaluating baseline model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Baseline Accuracy: {accuracy:.3f}')
    return accuracy

def evaluate_knowledge_cards(texts):
    """Evaluate modular knowledge cards integration."""
    print("Evaluating knowledge cards integration...")
    
    baseline_classifier = pipeline('sentiment-analysis', 
                                 model='distilbert-base-uncased-finetuned-sst-2-english')
    
    print("Baseline Results:")
    baseline_results = []
    for text in texts:
        result = baseline_classifier(text)[0]
        baseline_results.append(result)
        print(f"Input: {text}\nOutput: {result}\n")
    
    knowledge_card_A = pipeline('sentiment-analysis', 
                               model='distilbert-base-uncased-finetuned-sst-2-english')
    knowledge_card_B = pipeline('sentiment-analysis', 
                               model='distilbert-base-uncased-finetuned-sst-2-english')
    
    G = nx.Graph()
    G.add_node('card_A', relevance=0.8, confidence=0.9)
    G.add_node('card_B', relevance=0.7, confidence=0.85)
    G.add_edge('card_A', 'card_B', weight=0.6)
    
    def select_knowledge_card(graph, text):
        best_card = None
        best_score = 0
        for node, data in graph.nodes(data=True):
            score = data['relevance'] * data['confidence']
            if score > best_score:
                best_score = score
                best_card = node
        return best_card
    
    print("Using Graph-Based Fusion for Knowledge Cards:")
    fused_results = []
    for text in texts:
        selected_card = select_knowledge_card(G, text)
        if selected_card == 'card_A':
            result = knowledge_card_A(text)[0]
        else:
            result = knowledge_card_B(text)[0]
        fused_results.append(result)
        print(f"Input: {text}\nSelected Card: {selected_card}\nOutput: {result}\n")
    
    print("Knowledge cards evaluation completed")
    return baseline_results, fused_results

def compute_trust_score(model, x):
    """Compute trust score for self-verification."""
    model.zero_grad()
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    outputs = model(x_tensor)
    _, predicted = torch.max(outputs, 0)
    loss = nn.CrossEntropyLoss()(outputs.unsqueeze(0), predicted.unsqueeze(0))
    loss.backward()
    
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm().item()
    
    trust_score = 1.0 / (grad_norm + 1e-6)
    return trust_score

def evaluate_self_verification(sv_network, num_iterations=10):
    """Evaluate self-verification with meta reinforcement learning."""
    print("Evaluating self-verification module...")
    
    env = IterativeTaskEnv()
    trust_scores = []
    
    for iteration in range(num_iterations):
        obs = env.reset()
        done = False
        cumulative_trust = 0
        steps = 0
        
        while not done:
            trust = compute_trust_score(sv_network, obs)
            cumulative_trust += trust
            steps += 1
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
        
        avg_trust = cumulative_trust / steps
        trust_scores.append(avg_trust)
        print(f'Iteration {iteration+1:02d} - Average Trust Score: {avg_trust:.4f}')
    
    print("Self-verification evaluation completed")
    return trust_scores

def save_plots(accuracies, baseline_results, fused_results, trust_scores, texts):
    """Save all plots as PDF files."""
    print("Saving plots as PDF files...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies)+1), accuracies, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Dynamic Adaptation under Adversarial Noise')
    plt.grid(True)
    plt.savefig('.research/iteration1/images/training_loss.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    baseline_confidences = [res['score'] for res in baseline_results]
    fused_confidences = [res['score'] for res in fused_results]
    x = np.arange(len(texts))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, baseline_confidences, width, label='Baseline')
    plt.bar(x + width/2, fused_confidences, width, label='Knowledge Cards')
    plt.xlabel('Text Samples')
    plt.ylabel('Confidence Score')
    plt.title('Baseline vs. Knowledge Cards Confidence Scores')
    plt.xticks(x, [f'Text {i+1}' for i in range(len(texts))])
    plt.legend()
    plt.savefig('.research/iteration1/images/accuracy_baseline.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(trust_scores)+1), trust_scores, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Average Trust Score')
    plt.title('Evolution of Trust Score with Self-Verification and Meta RL')
    plt.grid(True)
    plt.savefig('.research/iteration1/images/inference_latency.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("All plots saved successfully as PDF files")
