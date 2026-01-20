public class Stats {
    public static void main(String[] args) {
        int[] numbers = {1,4,6,8,10};

        int total = 0;
        for (int x: numbers) {
            total += x;
        }

        double mean = (double) total / numbers.length;

        int countAboveMean = 0;
        for (int x: numbers) {
            if (x > mean) {
                countAboveMean++;
            }
        }

        System.out.println("Sum: " + total);
        System.out.println("Mean: " + mean);
        System.out.println("Count above mean: " + countAboveMean);            
    }
}

// run: javac compare_languages/java/Stats.java
// run: java -cp java Stats