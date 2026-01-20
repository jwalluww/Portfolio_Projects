numbers = [1,4,6,8,10]

total = sum(numbers)
mean = total / len(numbers)

count_above_mean = sum(1 for x in numbers if x > mean)

print(f"Sum: {total}")
print(f"Mean: {mean}")
print(f"Count above mean: {count_above_mean}")

# Run: python python/stats.py