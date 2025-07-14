import java.io.*;
import java.util.*;
import java.util.List;

public class GROUP_35_2021510110_2022510150_2022510184_java {

    public static class TreeNode {
        String attribute; // if this is an internal node
        String label;     // if this is a leaf node
        Map<String, TreeNode> children = new HashMap<>();

        // Constructor for internal node
        TreeNode(String attribute) {
            this.attribute = attribute;
        }

        // Constructor for leaf node
        TreeNode(String attribute, String label) {
            this.attribute = attribute;
            this.label = label;
        }

        boolean isLeaf() {
            return label != null;
        }
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String fileName ;
        String targetAttribute ;

        // Step 1: Dataset selection
        System.out.println("Select a dataset:");
        System.out.println("1 - Weather");
        System.out.println("2 - Breast Cancer");
        System.out.println("3 - Contact Lenses");
        System.out.print("Enter your choice (1/2/3): ");
        String choice = scanner.nextLine();

        switch (choice) {
            case "1":
                fileName = "weather.csv";
                targetAttribute = "play";
                break;
            case "2":
                fileName = "breast_cancer.csv";
                targetAttribute = "Class";
                break;
            case "3":
                fileName = "contact_lenses.csv";
                targetAttribute = "contact-lenses";
                break;
            default:
                System.out.println("Invalid choice.");
                return;
        }

        // Step 2: Load dataset
        List<Map<String, String>> dataset = loadDataset(fileName);

        if (dataset.isEmpty()) {
            System.out.println("Dataset is empty or not found.");
            return;
        }

        // Step 3: Prepare attributes
        List<String> attributes = new ArrayList<>(dataset.getFirst().keySet());
        attributes.remove(targetAttribute); // remove the label column

        // Step 4: Build decision tree
        TreeNode root = buildTree(dataset, attributes, targetAttribute);

        // Save the tree
        exportToDot(root, "tree_output.dot");
        generateImageFromDot("tree_output.dot", "output/tree_output.png");


        // Step 5: Prediction loop
        while (true) {
            System.out.println("\nEnter attribute values to predict (type 'exit' to quit):");

            Map<String, String> userInput = new HashMap<>();
            for (String attr : attributes) {
                System.out.print(attr + ": ");
                String value = scanner.nextLine().trim();

                if (value.equalsIgnoreCase("exit")) return;

                if (value.isEmpty()) {
                    System.out.println("Value cannot be empty. Please enter a valid value.");
                    return;
                }

                //Normalize value
                value = value.toLowerCase(Locale.ROOT).replaceAll("\\s+", "");

                userInput.put(attr, value);
            }

            String result = predict(root, userInput);
            System.out.println("Prediction: " + result);
        }
    }


    public static List<Map<String, String>> loadDataset(String fileName) {
        List<Map<String, String>> data = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String headerLine = br.readLine(); // First line: column names
            if (headerLine == null) return data;

            String[] headers = headerLine.split(","); // Split headers by comma

            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(","); // Split values by comma
                Map<String, String> row = new HashMap<>();
                for (int i = 0; i < headers.length; i++) {
                    row.put(headers[i].trim(), values[i].trim().toLowerCase());
                }
                data.add(row); // Add the row to the list
            }
        } catch (IOException e) {
            System.out.println("Error reading file: " + e.getMessage());
        }

        return data;
    }

    public static double calculateEntropy(List<Map<String, String>> records, String targetAttribute) {
        Map<String, Integer> classCounts = new HashMap<>();

        // Count how many times each output class appears
        for (Map<String, String> record : records) {
            String label = record.get(targetAttribute);  // e.g., "yes" or "no"
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        }

        double entropy = 0.0;
        int total = records.size();

        for (Map.Entry<String, Integer> entry : classCounts.entrySet()) {
            double p = (double) entry.getValue() / total;
            if (p > 0) {
                entropy -= p * (Math.log(p) / Math.log(2));
            }
        }

        return entropy;
    }

    public static double calculateInformationGain(List<Map<String, String>> dataset, String attribute, String targetAttribute) {
        double totalEntropy = calculateEntropy(dataset, targetAttribute);
        int totalSize = dataset.size();

        // Get unique values for the attribute (e.g., sunny, rainy)
        Set<String> attributeValues = new HashSet<>();
        for (Map<String, String> record : dataset) {
            attributeValues.add(record.get(attribute));
        }

        double weightedEntropy = 0.0;

        // For each value of the attribute, filter dataset and calculate entropy
        for (String value : attributeValues) {
            List<Map<String, String>> subset = new ArrayList<>();

            // Filter records where attribute == value
            for (Map<String, String> record : dataset) {
                if (record.get(attribute).equals(value)) {
                    subset.add(record);
                }
            }

            double subsetEntropy = calculateEntropy(subset, targetAttribute);
            double weight = (double) subset.size() / totalSize;
            weightedEntropy += weight * subsetEntropy;
        }
        return totalEntropy - weightedEntropy;
    }

    public static TreeNode buildTree(List<Map<String, String>> dataset, List<String> attributes, String targetAttribute) {
        // Get unique class labels (e.g., "yes", "no")
        Set<String> classLabels = new HashSet<>();
        for (Map<String, String> record : dataset) {
            classLabels.add(record.get(targetAttribute));
        }

        // Base case 1: All records have the same class label → return leaf node
        if (classLabels.size() == 1) {
            String label = classLabels.iterator().next();
            System.out.println("Leaf Node → All " + label);
            return new TreeNode(null, label); // Leaf node
        }

        // Base case 2: No more attributes to split → return majority class
        if (attributes.isEmpty()) {
            String majority = getMajorityClass(dataset, targetAttribute);
            System.out.println("Leaf Node → Majority Class: " + majority);
            return new TreeNode(null, majority);
        }

        // Step 1: Find best attribute by information gain
        //-------------------------------------------------
        String bestAttribute = null;
        double bestGain = -1;

        System.out.println("Subset:");
        System.out.println("- Records: " + dataset.size());
        Map<String, Integer> classCounts = new HashMap<>();
        for (Map<String, String> record : dataset) {
            String label = record.get(targetAttribute);
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        }
        classCounts.forEach((label, count) -> System.out.println("- " + targetAttribute + ": " + count + " " + label));
        System.out.println("Entropy ≈ " + calculateEntropy(dataset, targetAttribute));

        for (String attr : attributes) {
            double gain = calculateInformationGain(dataset, attr, targetAttribute);
            System.out.println(attr + " → Gain = " + gain);
            if (gain > bestGain) {
                bestGain = gain;
                bestAttribute = attr;
            }
        }

        if (bestAttribute == null) {
            String majority = getMajorityClass(dataset, targetAttribute);
            //terminal output
            System.out.println("Leaf Node → Majority Class: " + majority);

            return new TreeNode(null, majority);
        }

        System.out.println("Best Attribute: " + bestAttribute + " → Gain = " + bestGain);

        // Create internal node
        TreeNode node = new TreeNode(bestAttribute);

        // Step 2: For each value of bestAttribute, create a child
        Set<String> values = new HashSet<>();
        for (Map<String, String> record : dataset) {
            values.add(record.get(bestAttribute));
        }

        for (String value : values) {
            System.out.println("Split " + bestAttribute + " on " + value + ":");
            // Filter dataset where bestAttribute == value
            List<Map<String, String>> subset = new ArrayList<>();
            for (Map<String, String> record : dataset) {
                if (record.get(bestAttribute).equals(value)) {
                    subset.add(record);
                }
            }

            // Remove the used attribute from list for recursion
            List<String> remainingAttributes = new ArrayList<>(attributes);
            remainingAttributes.remove(bestAttribute);

            // Recursively build subtree
            TreeNode child = buildTree(subset, remainingAttributes, targetAttribute);
            node.children.put(value, child);
        }

        return node;
    }

    public static String getMajorityClass(List<Map<String, String>> dataset, String targetAttribute) {
        Map<String, Integer> countMap = new HashMap<>();
        for (Map<String, String> record : dataset) {
            String label = record.get(targetAttribute);
            countMap.put(label, countMap.getOrDefault(label, 0) + 1);
        }

        String majorityClass = null;
        int maxCount = -1;

        for (Map.Entry<String, Integer> entry : countMap.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                majorityClass = entry.getKey();
            }
        }

        return majorityClass;
    }

    public static String predict(TreeNode node, Map<String, String> input) {
        while (!node.isLeaf()) {
            String attr = node.attribute;
            String value = input.get(attr);

            TreeNode child = node.children.get(value);
            if (child == null) {
                return "Unknown (value '" + value + "' not seen in training)";
            }
            node = child;
        }

        return node.label;
    }

    //VISUALIZATION OF DECISION TREE
    //-------------------------------------------------------------------------------------------

    public static void exportToDot(TreeNode root, String filePath) {
        try (PrintWriter writer = new PrintWriter(filePath)) {
            writer.println("digraph DecisionTree {");
            writer.println("    node [shape=ellipse, style=filled, fillcolor = lightgoldenrod];");
            writeDotRecursive(writer, root);
            writer.println("}");
            System.out.println("DOT file written to: " + filePath);
        } catch (IOException e) {
            System.out.println("Failed to write DOT file: " + e.getMessage());
        }
    }

    private static int nodeCounter = 0;

    private static String writeDotRecursive(PrintWriter writer, TreeNode node) {
        String currentId = "node" + (nodeCounter++);
        String label = node.isLeaf() ? "Label: " + node.label : node.attribute;

        writer.printf("    %s [label=\"%s\"];\n", currentId, label);

        if (!node.isLeaf()) {
            for (Map.Entry<String, TreeNode> entry : node.children.entrySet()) {
                String edgeLabel = entry.getKey();
                String childId = writeDotRecursive(writer, entry.getValue());
                writer.printf("    %s -> %s [label=\"\", xlabel=\"%s\", fontcolor=red];\n", currentId, childId, edgeLabel);
            }
        }

        return currentId;
    }

    public static void generateImageFromDot(String dotFile, String outputPng) {
        try {
            File outputFile = new File(outputPng);
            File parentDir = outputFile.getParentFile();
            if (parentDir != null && !parentDir.exists()) {
                parentDir.mkdirs(); //create the directory if it doesn't exist
            }

            ProcessBuilder pb = new ProcessBuilder("dot", "-Tpng", dotFile, "-o", outputPng);
            pb.redirectErrorStream(true);
            Process process = pb.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                System.out.println("Tree image created: " + outputPng);
            } else {
                System.out.println("dot command failed.");
            }
        } catch (Exception e) {
            System.out.println("Graphviz generation error: " + e.getMessage());
        }
    }
}