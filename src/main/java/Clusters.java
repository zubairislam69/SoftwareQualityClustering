import java.io.File;

import net.sf.javaml.clustering.Clusterer;
import net.sf.javaml.clustering.KMeans;
import net.sf.javaml.clustering.evaluation.SumOfSquaredErrors;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;
import net.sf.javaml.clustering.evaluation.ClusterEvaluation;
import net.sf.javaml.clustering.FarthestFirst;
import net.sf.javaml.clustering.evaluation.TraceScatterMatrix;
import net.sf.javaml.clustering.evaluation.SumOfAveragePairwiseSimilarities;
import net.sf.javaml.clustering.KMedoids;

public class Clusters {

    public static void main(String[] args) throws Exception {
        // Load the iris dataset
        Dataset data = FileHandler.loadDataset(new File("C:src/main/resources/iris.data"), 4, ",");

        // Create a KMeans clusterer with k=3 clusters
        Clusterer km = new KMeans();

        Clusterer FarthestFirst = new FarthestFirst();

        Clusterer KMedoids = new KMedoids();


        //cluster the data using KMean and append to array
        Dataset[] kmClusters = km.cluster(data);
        System.out.println("KMeans Clusters:");
        printClusters(kmClusters);
        evaluateClusterer(km, data, kmClusters);
        System.out.println("----------------------------------------------");

        // Cluster the dataset using the AQBC algorithm
        Dataset[] FarthestFirstClusters = FarthestFirst.cluster(data);
        System.out.println("FarthestFirst Clusters:");
        printClusters(FarthestFirstClusters);
        evaluateClusterer(FarthestFirst, data, FarthestFirstClusters);
        System.out.println("----------------------------------------------");

        // Cluster the dataset using the DBSCAN algorithm
        Dataset[] KMedoidsClusters = KMedoids.cluster(data);
        System.out.println("KMedoids Clusters:");
        printClusters(KMedoidsClusters);
        evaluateClusterer(KMedoids, data, KMedoidsClusters);
        System.out.println("----------------------------------------------");
    }

    private static void printClusters(Dataset[] clusters) {
        for (int i = 0; i < clusters.length; i++) {
            System.out.println("Cluster " + i + ":");
            for (Instance instance : clusters[i]) {
                System.out.println(instance);
            }
            System.out.println();
        }
    }

    private static void evaluateClusterer(Clusterer clusterer, Dataset data, Dataset[] clusters) {
        ClusterEvaluation sse = new SumOfSquaredErrors();
        ClusterEvaluation tsm = new TraceScatterMatrix();
        ClusterEvaluation aps = new SumOfAveragePairwiseSimilarities();

        double sseScore = sse.score(clusters);
        double tsmScore = tsm.score(clusters);
        double apsScore = aps.score(clusters);

        System.out.println("Clusterer SumOfSquaredErrors score: " + sseScore);
        System.out.println("Clusterer TraceScatterMatrix score: " + tsmScore);
        System.out.println("Clusterer SumOfAveragePairwiseSimilarities score: " + apsScore);

    }

}