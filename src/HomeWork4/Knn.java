package HomeWork4;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;


public class Knn implements Classifier {
	
	public enum EditMode {None, Forwards, Backwards};
	private EditMode m_editMode = EditMode.None;
	public enum Majority {UNIFORM, WEIGHTED};
	public Majority maj;
	public int k,lp;
	private Instances m_trainingInstances , currentTrainingInstances ;
	private double distanceThreshold = Double.MAX_VALUE;
		


	public EditMode getEditMode() {
		return m_editMode;
	}

	public void setEditMode(EditMode editMode) {
		m_editMode = editMode;
	}

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		switch (m_editMode) {
		case None:
			noEdit(arg0);
			double bestWeightedError = Double.MAX_VALUE , bestUniformError = Double.MAX_VALUE , minError = Double.MAX_VALUE;
			int bestK = 0, bestP = 0;
			Majority bestMaj = Majority.UNIFORM;
			for(int i = 1; i <= 20 ; i++) {
				for(int p = 0; p < 4; p ++) {
					lp = p;
					k = i;
					//weighted majority
					maj = Majority.WEIGHTED;
					System.out.println("Combination "+ k + ", " + lp + ", " + maj);
					double currentError = crossValidationError(m_trainingInstances);
					if(currentError < bestWeightedError){
						bestWeightedError = currentError;
					}
					
					//uniform majority
					
					maj = Majority.UNIFORM;
					System.out.println("Combination "+ k + ", " + lp + ", " + maj);
					currentError = crossValidationError(m_trainingInstances);
					if(currentError < bestUniformError){
						bestUniformError = currentError;
					}
					
					if(minError > bestUniformError) {
						minError = bestUniformError;
						bestK = i;
						bestP = p;
						bestMaj = Majority.UNIFORM;
					}
					
					if(minError > bestWeightedError) {
						minError = bestWeightedError;
						bestK = i;
						bestP = p;
						bestMaj = Majority.WEIGHTED;
					}	
				}
			}
			k = bestK;
			lp = bestP;
			maj = bestMaj;
			System.out.println("Best combination: " + k + ", " + lp + ", " + maj);
			System.out.println("Best error: " + minError);
			currentTrainingInstances = m_trainingInstances;
			break;
		case Forwards:
			editedForward(arg0);
			break;
		case Backwards:
			editedBackward(arg0);
			break;
		default:
			noEdit(arg0);
			break;
		}
	}

	@Override
	public double classifyInstance(Instance instance) {
		Instances neighbors = findNearestNeighbors(instance);
		switch(maj) {
		case UNIFORM:
			return getClassVoteResult(neighbors);
		default:
			return getWeightedClassVoteResult(instance, neighbors);
		}
	}
	
	public double[] calcConfusion(Instances data) {
		double truePos=0, pos=0, falseNeg=0, neg=0;
		double[] res = new double[2];
		for(int i=0; i<data.numInstances(); i++) {
			if(classifyInstance(data.get(i)) == 0.0) { pos++;
				if(data.get(i).classValue() == 0.0) { truePos++; }
			}
			else { neg++;
				if(data.get(i).classValue() != 1.0) { falseNeg++; }
			}
		}
		res[0] = truePos / pos;
		res[1] = truePos / (truePos + falseNeg);
		return res;
	}
	
	private Instances findNearestNeighbors(Instance instance) {
		boolean newMax = true;
		Instances neighbors = new Instances(m_trainingInstances,0);
		for(int i=0; i<currentTrainingInstances.numInstances(); i++) {
			if(neighbors.numInstances() < k) {
				neighbors.add(currentTrainingInstances.get(i));
			}
			else {
				if(newMax) {
					Instance max = getMaxNeighbor(instance, neighbors);
					distanceThreshold = distance(instance, max);
					newMax = false;
				}
				if(distanceThreshold > distance(instance ,currentTrainingInstances.get(i))){
					neighbors.remove(neighbors.indexOf(max));
					neighbors.add(currentTrainingInstances.get(i));
					newMax = true;
				}
			}
		}
		return neighbors;
	}
	/**
	 * 
	 * @param a - instance to compare distance to.
	 * @param neighbors - group of instances to find furtherest neighbor.
	 * @return Furtherest instance friom a.
	 */
	
	private double getClassVoteResult (Instances neighbors) {
		int pos = 0;
		int neg = 0;
		for(int i=0; i<neighbors.numInstances(); i++) {
			if(neighbors.get(i).classValue() == 0) { pos++; }
			else { neg++; }
		}
		return (pos > neg) ? 0.0 : 1.0;
	}
	
	private double getWeightedClassVoteResult(Instance a , Instances neighbors){
		int pos = 0;
		int neg = 0;
		for(int i=0; i<neighbors.numInstances(); i++) {
			if(neighbors.get(i).classValue() == 0) { pos += (1.0/Math.pow(distance(a , neighbors.get(i)), 2)); }
			else { neg += (1.0/Math.pow(distance(a , neighbors.get(i)), 2)); }
		}
		return (pos > neg) ? 0.0 : 1.0;
		
	}
	
	
	
	private Instance getMaxNeighbor(Instance a, Instances neighbors) {
		Instance max = neighbors.get(0);
		for(int i=1; i<neighbors.numInstances(); i++) {
			if(distance(a,neighbors.get(i)) > distance(a,max)) {
				max = neighbors.get(i);
			}
		}
		return max;
	}
	private double distance(Instance a, Instance b) {
		if(lp == 0) return lInfinityDistance(a, b);
		return lPDistance(a, b); 
	}
	
	private double lPDistance(Instance a, Instance b) {
		double dist = 0;
		for(int i=0; i<a.numAttributes(); i++) {
			dist += Math.pow(Math.abs((a.value(i) - b.value(i))), lp);
			if(Math.pow(dist, 1.0/(double)lp) > distanceThreshold) {
				return Double.MAX_VALUE;
			}
		}
		return Math.pow(dist, 1.0/(double)lp);
	}
	
	private double lInfinityDistance(Instance a, Instance b) {
		double maxDist = 0;
		for(int i=0; i<a.numAttributes(); i++) {
			maxDist = Math.max(maxDist, Math.abs((a.value(i) - b.value(i))));
		}
		return maxDist;
	}
	
	private double calcAvgError(Instances data){
		double numErrors = 0;
		for(int i = 0; i < data.numInstances(); i++){
			if(classifyInstance(data.get(i)) != data.get(i).classValue()){
				numErrors++;
			}
		}
		return (numErrors / data.numInstances());
	}
	
	private double crossValidationError(Instances data) {
		double foldsAvgError = 0;
		for(int i = 0; i < 10; i ++){
			Instances fold = extractFold(10, m_trainingInstances);
			foldsAvgError += calcAvgError(fold);
		}
		return (foldsAvgError/10.0);
	}
	
	private Instances extractFold(int k , Instances data){
		data.randomize(new Random(1));
		Instances fold =  new Instances(m_trainingInstances , 0);
		currentTrainingInstances = new Instances(m_trainingInstances , 0);
		for(int i = 0; i < data.numInstances(); i++){
			if(i <  data.numInstances() / k){
				fold.add(data.get(i));
			}
			else{
				currentTrainingInstances.add(data.get(i));
			}
		}
		
		return fold;
		
	}
	
//	private Instances[] divideToKFolds(int k, Instances data) {
//		Instances[] folds = new Instances[k];
//		for(int i=0; i<k; i++) {
//			folds[i] = new Instances(data,0);
//		}
//		for(int i=0; i<data.numInstances(); i++) {
//			folds[i%k].add(data.get(i));
//		}
//		return folds;
//	}
	
//	public Instances getDataFolds(int exFold, Instances[] folds) {
//		Instances trainingData = new Instances(m_trainingInstances,0);
//		for(int i=0; i<folds.length; i++) {
//			if(i != exFold) {
//				for(int j=0; j<folds[i].numInstances(); j++) {
//					trainingData.add(folds[i].get(j));
//				}
//			}
//		}
//		return trainingData;
//	}

	private void editedForward(Instances instances) {
		//TODO: implement this method
	}

	private void editedBackward(Instances instances) {
		//TODO: implement this method
	}

	private void noEdit(Instances instances) {
		m_trainingInstances = new Instances(instances);
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}
}
