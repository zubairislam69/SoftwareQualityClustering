import java.io.File;
import net.sf.javaml.clustering.Clusterer;
import net.sf.javaml.clustering.KMeans;
import net.sf.javaml.clustering.evaluation.SumOfSquaredErrors;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.distance.DistanceMeasure;
import net.sf.javaml.tools.data.FileHandler;
import net.sf.javaml.clustering.evaluation.ClusterEvaluation;
import net.sf.javaml.clustering.FarthestFirst;
import net.sf.javaml.clustering.KMedoids;
import net.sf.javaml.clustering.evaluation.AICScore;
import net.sf.javaml.clustering.evaluation.Gamma;
import net.sf.javaml.clustering.evaluation.CIndex;
import net.sf.javaml.distance.EuclideanDistance;

public class Clusters {

    public static void main(String[] args) throws Exception {

        // Load the iris dataset
        Dataset data = FileHandler.loadDataset(new File("C:src/main/resources/iris.data"), 4, ",");

        //creates a KMeans cluster with 4 clusters
        Clusterer km = new KMeans();

        //creates a FarthestFirst cluster with 4 clusters
        Clusterer FarthestFirst = new FarthestFirst();

        //creates a KMedoids cluster with 4 clusters
        Clusterer KMedoids = new KMedoids();

        //cluster the data using KMean
        Dataset[] kmClusters = km.cluster(data);
        System.out.println("KMeans Clusters:");

        //call output function to show all clusters
        outputClusters(kmClusters);

        //evaluate the scores for each cluster
        evaluateClustererScores(kmClusters);
        System.out.println("----------------------------------------------");

        // Cluster the dataset using the FarthestFirst algorithm
        Dataset[] FarthestFirstClusters = FarthestFirst.cluster(data);
        System.out.println("FarthestFirst Clusters:");

        //call output function to show all clusters
        outputClusters(FarthestFirstClusters);

        //evaluate the scores for each cluster
        evaluateClustererScores(FarthestFirstClusters);
        System.out.println("----------------------------------------------");

        // Cluster the dataset using the KMediods algorithm
        Dataset[] KMedoidsClusters = KMedoids.cluster(data);
        System.out.println("KMedoids Clusters:");

        //call output function to show all clusters
        outputClusters(KMedoidsClusters);

        //evaluate the scores for each cluster
        evaluateClustererScores(KMedoidsClusters);
        System.out.println("----------------------------------------------");
    }

    //function to output the clusters inside the dataset
    private static void outputClusters(Dataset[] clusters) {

        //for the length of the dataset
        for (int i = 0; i < clusters.length; i++) {

            //print out each cluster heading
            System.out.println("Cluster " + i + ":");

            //print out each cluster value
            for (Instance instance : clusters[i]) {
                System.out.println(instance);
            }
            System.out.println();
        }
    }

    //function to evaluate the clusters scores inside the dataset
    private static void evaluateClustererScores(Dataset[] clusters) {

        //creates instance of EuclideanDistance
        DistanceMeasure dm = new EuclideanDistance();

        //initialize sum of squared errors evaluation
        ClusterEvaluation sse = new SumOfSquaredErrors();

        //initialize AICScore evaluation
        ClusterEvaluation aic = new AICScore();

        //initialize Gamma evaluation using EuclideanDistance
        ClusterEvaluation gamma = new Gamma(dm);

        //initialize CIndex evaluation using EuclideanDistance
        ClusterEvaluation cindex = new CIndex(dm);

        //calculate all the scores
        double sseScore = sse.score(clusters);
        double aicScore = aic.score(clusters);
        double gammaScore = gamma.score(clusters);
        double cindexScore = cindex.score(clusters);

        //print out all scores
        System.out.println("Clusterer SumOfSquaredErrors score: " + sseScore);
        System.out.println("Clusterer AICScore score: " + aicScore);
        System.out.println("Clusterer Gamma score: " + gammaScore);
        System.out.println("Clusterer CIndex score: " + cindexScore);

    }

}