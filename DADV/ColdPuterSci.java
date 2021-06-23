import java.util.*;

class ColdPuterSci {
	public static void main(String[] args) {
		int n;
		String str;
		Scanner sc1 = new Scanner(System.in);
		System.out.println("number of temperatures");
		n = sc1.nextInt();
		Scanner sc2 = new Scanner(System.in);
		System.out.println("temperatures:");
		str = sc2.nextLine();
		String[] temp = str.split(" ");

		for(int i = 1; i <= n; i++) {
			for (String t:temp){
				System.out.println(t);
			}
		}
	}
}