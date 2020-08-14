
public class CNN {

	public static void main(String[] args) {
		
		Integer[][] filter1 = new Integer[3][3];
		filter1[0] = new Integer[]{-1, 0, 1};
		filter1[1] = new Integer[]{-3, 0, 2};
		filter1[2] = new Integer[]{1, 1, 2};
		
		Integer[][] filter2 = new Integer[3][3];
		filter2[0] = new Integer[]{2, -2, 1};
		filter2[1] = new Integer[]{-1, 0, 2};
		filter2[2] = new Integer[]{3, -2, 0};
		
		Integer[][] input = new Integer[5][5];
		input[0] = new Integer[]{4, 4, 1, 3, 2};
		input[1] = new Integer[]{2, 2, 4, 1, 2};
		input[2] = new Integer[]{5, 1, 2, 5, 1};
		input[3] = new Integer[]{2, 1, 5, 2, 4};
		input[4] = new Integer[]{4, 3, 4, 5, 1};
		
		Integer[][] output1 = new Integer[3][3];
		Integer[][] output2 = new Integer[3][3];
		
		//each slot in output array
		System.out.println("Output 1:");
		for (int i = 0; i < filter1.length; i ++) {
			for (int j = 0; j < filter1.length; j ++) {
				int sum = 0;
				//filter iter
				for (int k = 0; k < filter1.length; k++) {
					for (int l = 0; l < filter1.length; l++) {
						int n = k + i;
						int m = l + j;
						sum = sum + filter1[k][l]*input[n][m];
					}
				}
				output1[i][j] = sum;
			}
		}
		
		for (int i = 0; i < output1.length; i ++) {
			for (int j = 0; j < output1.length; j ++) {
				System.out.print(output1[i][j] + " ");
			}
			System.out.println("");
		}
		
		//each slot in output array
		System.out.println("");
		System.out.println("Output 2:");
		for (int i = 0; i < filter1.length; i ++) {
			for (int j = 0; j < filter1.length; j ++) {
				int sum = 0;
				//filter iter
				for (int k = 0; k < filter2.length; k++) {
					for (int l = 0; l < filter2.length; l++) {
						int n = k + i;
						int m = l + j;
						sum = sum + filter2[k][l]*input[n][m];
					}
				}
				output2[i][j] = sum;
			}
		}
		
		for (int i = 0; i < output2.length; i ++) {
			for (int j = 0; j < output2.length; j ++) {
				System.out.print(output2[i][j] + " ");
			}
			System.out.println("");
		}
	}

}
