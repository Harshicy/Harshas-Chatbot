import matplotlib.pyplot as plt
import numpy as np

# Test data response times (in seconds)
response_times = [0.6, 0.7, 1.9, 3.5, 0.2, 2.8, 0.5, 0.3, 4.2, 3.0]

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(response_times, bins=np.arange(0, 5.5, 0.5), color='blue', edgecolor='black', alpha=0.7, label='Response Time Frequency')

# Add average line
avg_time = np.mean(response_times)
plt.axvline(avg_time, color='red', linestyle='--', label=f'Average: {avg_time:.1f}s')

# Add peak annotation
peak_time = max(response_times)
plt.annotate(f'Peak: {peak_time}s', xy=(peak_time, 2), xytext=(peak_time + 0.2, 2.5),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Labels and title
plt.title('Response Time Distribution of Harsha\'s Chatbot', fontsize=14)
plt.xlabel('Response Time (seconds)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.text(0.95, 0.95, '80% of responses < 3s', transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='top', horizontalalignment='right')

# Display the plot
plt.show()