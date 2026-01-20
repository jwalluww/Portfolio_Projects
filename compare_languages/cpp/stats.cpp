#include <iostream>
#include <vector>

int main() {
    std::vector<int> numbers = {1,4,6,8,10};

    int total = 0;
    for (int x : numbers) {
        total += x;
    }

    double mean = static_cast<double>(total) / numbers.size();

    int countAboveMean = 0;
    for (int x : numbers) {
        if (x > mean) {
            countAboveMean++;
        }
    }

    std::cout << "Sum: " << total << std::endl;
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Count above mean: " << countAboveMean << std::endl;

    return 0;
}

// run code: g++ cpp/stats.cpp -o cpp/stats
// run code: stats.exe