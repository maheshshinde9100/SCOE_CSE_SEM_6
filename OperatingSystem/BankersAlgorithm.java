import java.util.*;
public class BankersAlgorithm {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of processes: ");
        int n = sc.nextInt();
        System.out.print("Enter number of resources: ");
        int m = sc.nextInt();

        int[][] alloc = new int[n][m];
        int[][] max = new int[n][m];
        int[][] need = new int[n][m];
        int[] avail = new int[m];

        System.out.println("\nEnter Allocation Matrix:");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                alloc[i][j] = sc.nextInt();
            }
        }

        System.out.println("\nEnter Maximum Matrix:");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                max[i][j] = sc.nextInt();
            }
        }

        System.out.println("\nEnter Available Resources:");
        for (int j = 0; j < m; j++) {
            avail[j] = sc.nextInt();
        }

        // Calculate Need = Max - Allocation
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                need[i][j] = max[i][j] - alloc[i][j];
            }
        }

        boolean[] finish = new boolean[n];
        int[] safeSeq = new int[n];
        int[] work = Arrays.copyOf(avail, m);

        int count = 0;

        while (count < n) {
            boolean found = false;
            for (int i = 0; i < n; i++) {
                if (!finish[i]) {
                    int j;
                    for (j = 0; j < m; j++) {
                        if (need[i][j] > work[j])
                            break;
                    }
                    // If all needs are satisfied
                    if (j == m) {
                        for (int k = 0; k < m; k++) {
                            work[k] += alloc[i][k];
                        }

                        safeSeq[count++] = i;
                        finish[i] = true;
                        found = true;
                    }
                }
            }

            if (!found) {
                System.out.println("\nSystem is NOT in safe state!");
                return;
            }
        }

        System.out.println("\nSystem is in SAFE state.");
        System.out.print("Safe Sequence: ");
        for (int i = 0; i < n; i++) {
            System.out.print("P" + safeSeq[i] + " ");
        }
    }
}