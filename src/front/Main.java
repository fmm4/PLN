package front;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Set;
import java.util.Vector;

import org.apache.commons.codec.binary.StringUtils;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.tartarus.snowball.ext.PorterStemmer;

import me.jhenrique.manager.TweetManager;
import me.jhenrique.manager.TwitterCriteria;
import me.jhenrique.model.Tweet;
import back.SentiWordNet;
import back.TwitterController;
import back.neuralNetwork;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.pmml.jaxbbindings.SupportVectorMachine;
import weka.core.stemmers.SnowballStemmer;
import weka.filters.unsupervised.*;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Main {

	private static TwitterController tc;
	private static Set<Object> stopSet = EnglishAnalyzer.getDefaultStopSet();
	private static SentiWordNet sent;

	public static void main(String[] args) throws Exception {
		try {
			sent = new SentiWordNet("Sentiwordnet/SentiWordNet_3.0.0.txt");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		String dataFolder = "Dados Classificados/classificados/";

		Vector<Integer> values = new Vector<Integer>();

		values = loadCSV(dataFolder);

		Vector<String> filenames = listFilesForFolder(new File(dataFolder));

		Vector<String[]> classified = classify(filenames,dataFolder,values);


		NaiveBayes nb = new NaiveBayes();	
		NaiveBayesMultinomial nbm = new NaiveBayesMultinomial();
		RandomForest rf = new RandomForest();
		SMO svm = new SMO();

		MessageClassifier nbmNonSupervi = new MessageClassifier(nbm);
		String unsupDataFolder = "unsurpervised/";

		Vector<String[]> supportHillary = readFolder(230+20,unsupDataFolder+"imwithher/","Hillary");
		supportHillary.addAll( readFolder(230+20,unsupDataFolder+"nevertrump/","Hillary"));

		Vector<String[]> supportTrump = readFolder(230+20,unsupDataFolder+"MakeAmericaGreatAgain/","Trump");
		supportTrump.addAll(readFolder(230+20,unsupDataFolder+"neverhillary/","Trump"));

		for(String[] s:supportHillary)
		{
			nbmNonSupervi.updateData(s[0], "Hillary");
		}

		for(String[] s:supportTrump)
		{
			nbmNonSupervi.updateData(s[0], "Trump");
		}		
		Vector<String[]> unclassifiedData = new Vector<String[]>();
		unclassifiedData.addAll(supportTrump);
		unclassifiedData.addAll(supportHillary);

		double nbPrecision = 0;
		double nbmPrecision = 0;
		double rfPrecision = 0;
		double svmPrecision = 0;
		double nbRecall = 0;
		double nbmRecall = 0;
		double svmRecall = 0;
		double rfRecall = 0;
		double lbRecall = 0;
		double lbPrecision = 0;
		double caRecall = 0;
		double caPrecision = 0;
		double cpRecall = 0;
		double cpPrecision = 0;
		double clRecall = 0;
		double clPrecision = 0;
		
		Vector<String> resultadosCA = new Vector<String>();
		Vector<String> resultadosCP = new Vector<String>();
		
		Vector<String> nbResul = new Vector<String>();
		Vector<String> nbmResul = new Vector<String>();
		Vector<String> rfResul = new Vector<String>();
		Vector<String> ldResul = new Vector<String>();
		Vector<String> svmResul = new Vector<String>();
		
		

		Vector<Vector<String[]> > particoes = splitTraining(2+3,classified);
		for(int i = 0; i < 2+3; i++)
		{
			//System.out.println("Teste "+i+":");
			Vector<String[]> test = particoes.elementAt(i);
			Vector<String[]> training = new Vector<String[]>();
			for(int o = 0; o < 2+3; o++)
			{
				if(o!=i)
				{
					for(String[] s:particoes.elementAt(o))
					{						
						training.add(s);
					}
				}
			}
			Vector<Vector<String> > artComiteeResults = new Vector<Vector<String> >();

			Vector<String> nbResults = getResult(nb,training,test);
			Vector<String> svmResults = getResult(svm,training,test);
			Vector<String> rfResults = getResult(rf,training,test);
			Vector<String> nbmResults = getResult(nbmNonSupervi,test);
			Vector<String> ldResults = getResultsSentiment(test);
			artComiteeResults.add(nbResults);
			artComiteeResults.add(nbmResults);
			artComiteeResults.add(ldResults);			
			Vector<String> cmResults = getComiteeResult(artComiteeResults);
			resultadosCP.addAll(cmResults);
			artComiteeResults.add(rfResults);
			artComiteeResults.add(svmResults);
			Vector<String> acmResults = getComiteeResult(artComiteeResults);
			artComiteeResults.remove(2);
			Vector<String> lcmResults = getComiteeResult(artComiteeResults);
			resultadosCA.addAll(acmResults);
			
			nbResul.addAll(nbResults);
			nbmResul.addAll(nbmResults);
			rfResul.addAll(rfResults);
			ldResul.addAll(ldResults);
			svmResul.addAll(svmResults);
			

			double[][] confMatrixNB = getConfusionMatrix(test,nbResults);
			double[][] confMatrixNBM = getConfusionMatrix(test,nbmResults);
			double[][] confMatrixRF = getConfusionMatrix(test,rfResults);
			double[][] confMatrixSVM = getConfusionMatrix(test,svmResults);
			double[][] confMatrixLD = getConfusionMatrix(test,ldResults);
			double[][] confMatrixCP = getConfusionMatrix(test,cmResults);
			double[][] confMatrixCA = getConfusionMatrix(test,acmResults);
			double[][] confMatrixCL = getConfusionMatrix(test,lcmResults);


			nbPrecision += getPrecision(confMatrixNB);
			nbmPrecision += getPrecision(confMatrixNBM);
			nbRecall += getRecall(confMatrixNB);
			nbmRecall += getRecall(confMatrixNBM);
			rfPrecision += getPrecision(confMatrixRF);
			svmPrecision += getPrecision(confMatrixSVM);
			rfRecall += getRecall(confMatrixRF);
			svmRecall += getRecall(confMatrixSVM);
			lbRecall += getRecall(confMatrixLD);
			lbPrecision += getPrecision(confMatrixLD);
			cpRecall += getRecall(confMatrixCP);
			cpPrecision += getPrecision(confMatrixCP);
			caRecall += getRecall(confMatrixCA);
			caPrecision += getPrecision(confMatrixCA);
			clRecall += getRecall(confMatrixCL);
			clPrecision += getPrecision(confMatrixCL);
		}

		System.out.println("Precisao NB: "+nbPrecision/(2+3)+" Recall NB:"+nbRecall/(2+3)+" FMeasure: "+fmeasure(nbPrecision,nbRecall)/(2+3));
		System.out.println("Precisao NBM: "+nbmPrecision/(2+3)+" Cobertura NBM:"+nbmRecall/(2+3)+" FMeasure: "+fmeasure(nbmPrecision,nbmRecall)/(2+3));
		System.out.println("Precisao RF: "+rfPrecision/(2+3)+" Recall RF:"+rfRecall/(2+3)+" FMeasure: "+fmeasure(rfPrecision,rfRecall)/(2+3));
		System.out.println("Precisao SVM: "+svmPrecision/(2+3)+" Recall SVM:"+svmRecall/(2+3)+" FMeasure: "+fmeasure(svmPrecision,svmRecall)/(2+3));
		System.out.println("Precisao LD: "+lbPrecision/(2+3)+" Recall LD:"+lbRecall/(2+3)+" FMeasure: "+fmeasure(lbPrecision,lbRecall)/(2+3));
		System.out.println("Precisao CP: "+cpPrecision/(2+3)+" Recall CP:"+cpRecall/(2+3)+" FMeasure: "+fmeasure(cpPrecision,cpRecall)/(2+3));
		System.out.println("Precisao CA: "+caPrecision/(2+3)+" Recall CA:"+caRecall/(2+3)+" FMeasure: "+fmeasure(caPrecision,caRecall)/(2+3));
		System.out.println("Precisao CL: "+clPrecision/(2+3)+" Recall CL:"+clRecall/(2+3)+" FMeasure: "+fmeasure(clPrecision,clRecall)/(2+3));
		
		System.out.println("'Resultado' Eleicao:");
		System.out.println("Previsao suposta:");
		resultadoEleicao(classified);
		System.out.println("Previsao suposta:");
		resultadoEleicao(classified);
		System.out.println("Previsao Argigo:");
		resultadoEleicao(classified, resultadosCP);
		System.out.println("Previsao Modificada:");
		resultadoEleicao(classified, resultadosCA);
		
		Vector<String> testNeural = new Vector<String>();
		
		for(String[] k:classified)
		{
			testNeural.add(k[1]);
		}

		double nnRecall = 0;
		double nnPrecision = 0;
		Vector<Vector<String>> particoesNeural = splitTraining2(2+3,testNeural);
		for(int i = 0; i < 2+3; i++)
		{
			
			//System.out.println("Teste "+i+":");
			Vector<String> test = particoesNeural.elementAt(i);
			Vector<String> training = new Vector<String>();
			for(int o = 0; o < 2+3; o++)
			{
				if(o!=i)
				{
					training.addAll(particoesNeural.elementAt(o));
				}
			}
			neuralNetwork nr = new neuralNetwork();
			
			for(int o = 0; o < 2+3; o++)
			{
				if(o!=i)
				{
					for(int c = o*40; c < o*40+40; c++)
					{
						nr.updateData(nbResul.elementAt(c),
								nbmResul.elementAt(c),
								rfResul.elementAt(c),
								svmResul.elementAt(c),
								ldResul.elementAt(c),
								test.elementAt(c-(o*40))
								);				
					}
				}
			}
			
			Vector<String> resultTest = new Vector<String>();
			for(int c = i*40; c < i*40+40; c++)
			{
				resultTest.add(nr.classifyMessage(nbResul.elementAt(c),
						nbmResul.elementAt(c),
						rfResul.elementAt(c),
						svmResul.elementAt(c),
						ldResul.elementAt(c)
						));				
			}			

			double[][] confMatrixNN = getConfusionMatrix2(test,resultTest);

			nnPrecision += getPrecision(confMatrixNN);
			nnRecall += getRecall(confMatrixNN);
		}
		System.out.println("Precisao NN: "+nnPrecision/(2+3)+" Recall NN:"+nnRecall/(2+3)+" FMeasure: "+fmeasure(nnPrecision,nnRecall)/(2+3));
	}
	

	
	public static double fmeasure(double p,double r)
	{
		return 2*(p*r)/(p+r);
	}
	
	public static Vector<String>  getComiteeResult(Vector<Vector<String> > test)
	{
		Vector<String> comiteeResults = new Vector<String>();
		for(int i = 0; i < test.elementAt(0).size(); i++)
		{
			comiteeResults.add(votingResults(test,i));			
		}
		return comiteeResults;
	}
	
	public static String votingResults(Vector<Vector<String> > test, int i)
	{
		int result = 0;
		for(Vector<String> k:test)
		{
			if(k.elementAt(i).equals("Hillary"))
			{
				result++;			
			}else{
				result--;
			}
		}
		if(result>0)
		{
			return "Hillary";
		}else{
			return "Trump";
		}
	}
	
	public static void resultadoEleicao(Vector<String[]> test, Vector<String> predicted)
	{
		double hillary = 0;
		double trump = 0;
		for(int i = 0; i < test.size(); i++)
		{
			if(predicted.elementAt(i).equals("Hillary"))
			{
				hillary+= 1.0/(double) Integer.parseInt(test.elementAt(i)[2]);
			}
			if(predicted.elementAt(i).equals("Trump"))
			{
				trump+= 1.0/(double) Integer.parseInt(test.elementAt(i)[2]);
			}
		}
		System.out.println("HillaryScore: "+hillary+" TrumpScore:"+trump);
	}
	
	public static void resultadoEleicao(Vector<String[]> test)
	{
		double hillary = 0;
		double trump = 0;
		for(int i = 0; i < test.size(); i++)
		{
			if(test.elementAt(i)[1].equals("Hillary"))
			{
				hillary+= 1.0/(double) Integer.parseInt(test.elementAt(i)[2]);
			}
			if(test.elementAt(i)[1].equals("Trump"))
			{
				trump+= 1.0/(double) Integer.parseInt(test.elementAt(i)[2]);
			}
		}
		System.out.println("HillaryScore: "+hillary+" TrumpScore:"+trump);
	}
	
	public static Vector<String> getResultsSentiment(Vector<String[]> test)
	{
		Vector<String> results = new Vector<String>();
		for(String[] g:test)
		{
			//System.out.println("------------------------");
			String resultado = getSupport(g[0]);
			results.add(resultado);	
			//System.out.println("Real: "+g[1]);
			//System.out.println("Classified: "+resultado);
			if(!g[1].equals(resultado))
			{
				//System.out.println("Tweet: "+g[0]);
			}
		}
		
		return results;
	}

	public static void printMatrix(double[][] k)
	{
		for(int i = 0; i < k.length;i++ )
		{
			for(int o = 0; o < k[i].length; o++)
			{
				System.out.print(k[i][o]+"	");
			}
			System.out.println();
		}
	}

	public static Vector<String> getResult(MessageClassifier classifier, Vector<String[]> setTest) throws Exception{
		MessageClassifier nbClass = classifier;

		Vector<String> results = new Vector<String>();
		for(String[] s:setTest)
		{
			results.add(nbClass.classifyMessage(s[0]));
		}

		return results;
	}

	public static Vector<String> getResult(Classifier classifier, Vector<String[]> setTrain, Vector<String[]> setTest) throws Exception{
		MessageClassifier nbClass = new MessageClassifier(classifier);
		for(String[] s:setTrain)
		{
			nbClass.updateData(s[0],s[1]);
		}

		Vector<String> results = new Vector<String>();
		for(String[] s:setTest)
		{
			results.add(nbClass.classifyMessage(s[0]));
		}

		return results;
	}

	public static double[][] getConfusionMatrix(Vector<String[]> test,Vector<String> result)
	{
		double[][] confMatrix = new double[2][2];
		for(int i = 0; i < test.size(); i++)
		{
			if(result.elementAt(i).equals(test.elementAt(i)[1]))
			{
				if(result.elementAt(i).equals("Trump"))
				{
					confMatrix[0][0]+=1;
				}
				if(result.elementAt(i).equals("Hillary"))
				{
					confMatrix[1][1]+=1;
				}
			}
			if(!result.elementAt(i).equals(test.elementAt(i)[1]))
			{
				if(result.elementAt(i).equals("Hillary"))
				{
					confMatrix[0][1]+=1;
				}
				if(result.elementAt(i).equals("Trump"))
				{
					confMatrix[1][0]+=1;
				}
			}
		}
		return confMatrix;
	}
	
	public static double[][] getConfusionMatrix2(Vector<String> test,Vector<String> result)
	{
		double[][] confMatrix = new double[2][2];
		for(int i = 0; i < test.size(); i++)
		{
			if(result.elementAt(i).equals(test.elementAt(i)))
			{
				if(result.elementAt(i).equals("Trump"))
				{
					confMatrix[0][0]+=1;
				}
				if(result.elementAt(i).equals("Hillary"))
				{
					confMatrix[1][1]+=1;
				}
			}
			if(!result.elementAt(i).equals(test.elementAt(i)))
			{
				if(result.elementAt(i).equals("Hillary"))
				{
					confMatrix[0][1]+=1;
				}
				if(result.elementAt(i).equals("Trump"))
				{
					confMatrix[1][0]+=1;
				}
			}
		}
		return confMatrix;
	}

	public static double getPrecision(double[][] k)
	{
		double total = k[0][0]+k[0][1]+k[1][0]+k[1][1];
		double right = k[0][0]+k[1][1];
		return right/total;		
	}

	public static double getRecall(double[][] k)
	{
		double trumpRec = k[0][0]/(k[0][0]+k[0][1]);
		double hillaryRec = k[1][1]/(k[1][1]+k[1][0]);
		return (trumpRec+hillaryRec)/2;

	} 	

	public static Vector<Vector<String[]> > splitTraining(int number, Vector<String[]> stuff)
	{
		int k = stuff.size()/number;
		Vector<Vector<String[]> > vector = new Vector<Vector<String[]> >();
		int f = 0;
		for(int i = 0; i < number; i++)
		{
			Vector<String[]> vec = new Vector<String[]>();
			for(int o = 0; o < k; o++)
			{
				vec.add(stuff.elementAt(o+i*k));
				f++;
			}
			vector.add(vec);
		}
		return vector;		
	}
	
	public static Vector<Vector<String>> splitTraining2(int number, Vector<String> resultadosCA)
	{
		int k = resultadosCA.size()/number;
		Vector<Vector<String> > vector = new Vector<Vector<String> >();
		int f = 0;
		for(int i = 0; i < number; i++)
		{
			Vector<String> vec = new Vector<String>();
			for(int o = 0; o < k; o++)
			{
				vec.add(resultadosCA.elementAt(o+i*k));
				f++;
			}
			vector.add(vec);
		}
		return vector;		
	}




	public static Vector<String[]> classify(Vector<String> filenames, String folder,Vector<Integer> k) throws IOException
	{
		Vector<String[]> returned = new Vector<String[]>();
		for(int i = 0; i < filenames.size(); i++)
		{
			String[] tempString = new String[3];
			String temp = read(folder+filenames.elementAt(i));
			String weight = filenames.elementAt(i).split("-")[0];
			tempString[2] = weight;
			tempString[0] = temp;

			String candidate;
			if((Integer) k.elementAt(i)==0)
			{
				candidate = "Trump";
			}else{
				candidate = "Hillary";
			}
			tempString[1] = candidate;
			returned.add(tempString);
		}
		return returned;
	}



	public static Vector<String> listFilesForFolder(final File folder) {
		Vector<String> filenames = new Vector<String>();
		for (final File fileEntry : folder.listFiles()) {
			if (fileEntry.isDirectory()) {
				listFilesForFolder(fileEntry);
			} else if(fileEntry.getName().contains(".txt")){
				filenames.addElement(fileEntry.getName());
			}
		}
		Collections.sort(filenames, new Comparator<String>() {

			public int compare(String o1, String o2) {
				// TODO tweak the comparator here 
				try{
					String[] split1 = o1.split("-");
					String[] split2 = o2.split("-");
					Integer integer1 = Integer.valueOf(split1[0]);
					Integer integer2 = Integer.valueOf(split2[0]);
					if(integer1!=integer2){
						return integer1.compareTo(integer2);
					}else{
						String[] fsplit1 = split1[1].split(".txt");
						String[] fsplit2 = split2[1].split(".txt");
						Integer finteger1 = Integer.valueOf(fsplit1[0]);
						Integer finteger2 = Integer.valueOf(fsplit2[0]);
						return finteger1.compareTo(finteger2);
					}
				}catch (java.lang.NumberFormatException e) {
					return o1.compareTo(o2);
				}
			}
		});
		return filenames;
	}


	public static Vector<Integer> loadCSV(String folder) throws IOException
	{
		BufferedReader br = new BufferedReader(new FileReader(folder+"classified.csv"));
		String line = br.readLine();
		String[] results = line.split(",");
		Vector<Integer> values = new Vector<Integer>();
		for(String s:results)
		{
			values.add(Integer.parseInt(s));
		}
		return values;
	}

	public static String read(String filename) throws IOException
	{
		BufferedReader in = new BufferedReader(new FileReader(filename));
		String line = in.readLine();
		in.close();
		return line;
	}


	public static Vector<String> vectorize(String string)
	{
		String[] splits = string.split("\\s+");	
		Vector<String> returned = new Vector<String>();
		for(String k:splits)
		{
			if(!stopSet.contains(k))
			{
				returned.add(stemTerm(k));
			}
		}
		return returned;
	}

	public static String stemTerm (String term) {
		PorterStemmer stem = new PorterStemmer();
		stem.setCurrent(term);
		stem.stem();
		String result = stem.getCurrent();
		return result;
	}


	public static String nN(int k)
	{
		String returned = "";
		if(k<10)
		{
			returned = "0"+k;
		}else{
			returned += k;
		}
		return returned;
	}

	public static boolean isValid(String p)
	{
		if((p.toLowerCase().contains("hillary") || p.toLowerCase().contains("clinton") || p.toLowerCase().contains("dem")) || (p.toLowerCase().contains("trump") || p.toLowerCase().contains("donald") || p.toLowerCase().contains("rep")))
		{
			return true;
		}
		return false;
	}

	public static Vector<String> extractingTweet(int n,String stringToSearch,String day,String month,String finalday,String finalmonth){
		TwitterCriteria criteria = null;
		List<Tweet> t = null;

		System.out.print(stringToSearch+" [2016-"+month+"-"+day+" to ");
		System.out.println("2016-"+finalmonth+"-"+finalday+"]");

		criteria = TwitterCriteria.create()
				.setQuerySearch(stringToSearch)
				.setSince("2016-"+month+"-"+day)
				.setUntil("2016-"+finalmonth+"-"+finalday)
				.setMaxTweets(n);
		t = TweetManager.getTweets(criteria);

		Vector<String> retorno = new Vector<String>();


		for(Tweet t1:t)
		{
			//System.out.println(t1.getText());
			retorno.addElement(t1.getText());
		}

		return retorno;
	}

	public static Vector<String[]> readFolder(int max, String folder, String string1) throws IOException
	{
		Vector<String[]> returned = new Vector<String[]>();
		for(int i = 0; i < max; i++)
		{
			String[] tempString = new String[2];
			tempString[0] = read(folder+"/"+(i+1)+".txt");
			tempString[1] = string1;
			returned.add(tempString);
		}
		return returned;
	}


	public static void writeToFile(int tweetn, int k, String tweetcontent)
	{
		try{
			PrintWriter writer = new PrintWriter(k+"-"+tweetn+".txt", "UTF-8");
			writer.println(tweetcontent);
			writer.close();
		} catch (IOException e) {
			//System.out.println(e);
			// do something
		}	
	}

	public static void writeToFile(int k, String tweetcontent,String hashtag)
	{
		new File(hashtag).mkdirs();
		try{
			PrintWriter writer = new PrintWriter(hashtag+"/"+k+".txt", "UTF-8");
			writer.println(tweetcontent);
			writer.close();
		} catch (IOException e) {
			System.out.println(e);
			// do something
		}	
	}

	public static String getSupport(String a)
	{
		a = a.toLowerCase();
		double trumpvotes = 0;
		double hillaryvotes = 0;
		double mentionTrump = 0;
		double mentionHillary = 0;
		mentionHillary += count(a,"hillary");
		mentionHillary += count(a,"clinton");
		mentionTrump += count(a,"donald");
		mentionTrump += count(a,"trump");
		mentionTrump += count(a,"mike");
		mentionTrump += count(a,"pence");

		trumpvotes += trumpTags(a);
		hillaryvotes += hillaryTags(a);
		
		double getOverrallSentiment = phraseSentiment(a);
		
		double feelinghillary = hillaryTags(a);
		double feelingtrump = trumpTags(a);
		
		if(feelingtrump>feelinghillary)
		{
			//System.out.println(feelingtrump +" > "+feelinghillary);
			//System.out.println("Detected important Trump hashtag");
			return "Trump";
		}else if(feelinghillary>feelingtrump){
			//System.out.println(feelingtrump +" < "+feelinghillary);
			//System.out.println("Detected important Hillary hashtag");
			return "Hillary";
		}
		//System.out.println("Using sentiment");

		String target;
		
		if(mentionHillary>=mentionTrump)
		{
			if(getOverrallSentiment>0)
			{
				return "Hillary";
			}else{
				return "Trump";
			}
		}else{
			if(getOverrallSentiment>0)
			{
				return "Trump";
			}else{
				return "Hillary";
			}
		}
		
	}
	
	public static double phraseSentiment(String a)
	{
		String[] splits = a.split("\\s+");
		double k = 0;
		for(String g:splits)
		{
			//System.out.println(g);
			k+=sent.extract(g);
		}
		return k;
	}


	public static double hillaryTags(String k)
	{
		double returned = 0;
		String[] hashtagList = {"#imwithher","#votedem","drumpf","racist","sexist","nevertrump","billionaire"};
		for(String s:hashtagList)
		{
			if(k.contains(s))
			{
				returned++;
				//System.out.println("Contains: "+s);
			}
		}
		return returned;
	}
	
	public static double trumpTags(String k)
	{
		double returned = 0;
		String[] hashtagList = {"#makeamericagreatagain","#voterep","shillary","killary","corrupt","neverhillary"};
		for(String s:hashtagList)
		{
			if(k.contains(s))
			{
				returned++;
				//System.out.println("Contains: "+s);
			}
		}
		return returned;
	}

	public static int count(final String string, final String substring)
	{
		int count = 0;
		int idx = 0;

		while ((idx = string.indexOf(substring, idx)) != -1)
		{
			idx++;
			count++;
		}

		return count;
	}


	//deprecated//

	//Extractor
	//Extrai os tweets, usado para achar antes da eleicao
	//			Calendar cal = Calendar.getInstance();
	//			cal.set(2016, 10, 7); //DIA DA ELEI플O
	//			System.out.println(cal.getTime().getDate());
	//	
	//			
	//			Calendar cal2 = cal;
	//			
	//			Vector<Vector<String> > tweets = new Vector<Vector<String> >();
	//			
	//			String year = "2016";
	//			for(int i = 0; i < 14; i++)
	//			{
	//				Date t = cal.getTime();
	//				int finalday,finalmonth;
	//				int firstday,firstmonth;
	//				finalday = t.getDate();
	//				finalmonth = t.getMonth();
	//				cal.add(Calendar.DAY_OF_YEAR, -7);
	//				t = cal.getTime();
	//				firstday = t.getDate();
	//				firstmonth = t.getMonth();
	//				Vector<String> twts = extractingTweet(300,"#election2016",nN(firstday),nN(firstmonth),nN(finalday),nN(finalmonth));
	//				System.out.println(twts.size());			
	//				tweets.add(twts);
	//				
	//			}
	//			System.out.println(tweets.size());
	//			int tweetn = 0;
	//			for(int p = 0; p < tweets.size(); p++)
	//			{
	//				Collections.shuffle(tweets.elementAt(p));
	//				int k = 0;
	//				for(String s:tweets.elementAt(p))
	//				{
	//					if(!s.contains("http") && !s.contains("RT") && isValid(s) && !s.contains("pic") && !s.contains(".com") && !s.contains("news"))
	//					{
	//						k++;
	//						tweetn++;
	//						writeToFile(tweetn,p,s);
	//					}
	//					if(k>=100)
	//					{
	//						System.out.println(p+": Success! 23+2");
	//						break;
	//					}
	//				}
	//				System.out.println(k);
	//				if(k==0)
	//				{
	//					for(String s:tweets.elementAt(p))
	//					{
	//						if(!s.contains("http") && !s.contains("RT"))
	//						{
	//							System.out.println(s);
	//						}
	//					}
	//				}
	//			}
	//		Calendar cal = Calendar.getInstance();
	//		cal.set(2016, 10, 7); //DIA DA ELEI플O
	//		int final_day = cal.getTime().getDate();
	//		int final_month = cal.getTime().getMonth()+1;
	//		cal.add(Calendar.WEEK_OF_YEAR,-14);
	//		int first_day = cal.getTime().getDate();
	//		int first_month = cal.getTime().getMonth()+1;
	//		String hashtag = "#imwithher";
	//	
	//		Vector<String> p = extractingTweet(3000,hashtag,nN(first_day),nN(first_month),nN(final_day),nN(final_month));
	//		System.out.println(p.size());
	//		int limit = 0;	
	//		for(int i = 0; i < p.size(); i++)
	//		{
	//			String s = p.elementAt(i);
	//			//System.out.println(s);
	//			if(!s.contains("http") && !s.contains("RT") && isValid(s) && !s.contains("pic") && !s.contains(".com") && !s.contains("news"))
	//			{
	//				limit++;
	//				writeToFile(limit,s,hashtag);
	//			}
	//			if(limit>250)
	//			{
	//				System.out.println("Success! 250");
	//				break;
	//			}
	//		}
	//		}

}

//		Calendar cal = Calendar.getInstance();
//		cal.set(2016, 10, 7); //DIA DA ELEI플O
//		System.out.println(cal.getTime().getDate());
//
//		
//		Calendar cal2 = cal;
//		
//		Vector<Vector<String> > tweets = new Vector<Vector<String> >();
//		
//		String year = "2016";
//		for(int i = 0; i < 14; i++)
//		{
//			Date t = cal.getTime();
//			int finalday,finalmonth;
//			int firstday,firstmonth;
//			finalday = t.getDate();
//			finalmonth = t.getMonth();
//			cal.add(Calendar.DAY_OF_YEAR, -7);
//			t = cal.getTime();
//			firstday = t.getDate();
//			firstmonth = t.getMonth();
//			Vector<String> twts = extractingTweet(300,"#election2016",nN(firstday),nN(firstmonth),nN(finalday),nN(finalmonth));
//			System.out.println(twts.size());			
//			tweets.add(twts);
//			
//		}
//		System.out.println(tweets.size());
//		int tweetn = 0;
//		for(int p = 0; p < tweets.size(); p++)
//		{
//			Collections.shuffle(tweets.elementAt(p));
//			int k = 0;
//			for(String s:tweets.elementAt(p))
//			{
//				if(!s.contains("http") && !s.contains("RT") && isValid(s) && !s.contains("pic") && !s.contains(".com") && !s.contains("news"))
//				{
//					k++;
//					tweetn++;
//					writeToFile(tweetn,p,s);
//				}
//				if(k>=100)
//				{
//					System.out.println(p+": Success! 23+2");
//					break;
//				}
//			}
//			System.out.println(k);
//			if(k==0)
//			{
//				for(String s:tweets.elementAt(p))
//				{
//					if(!s.contains("http") && !s.contains("RT"))
//					{
//						System.out.println(s);
//					}
//				}
//			}
//		}
//Calendar cal1 = Calendar.getInstance();
//cal1.set(2016, 10, 7); //DIA DA ELEI플O
//int final_day = cal1.getTime().getDate();
//int final_month = cal1.getTime().getMonth()+1;
//cal1.add(Calendar.WEEK_OF_YEAR,-14);
//int first_day = cal1.getTime().getDate();
//int first_month = cal1.getTime().getMonth()+1;
//String hashtag = "#neverhillary";
//
//Vector<String> p = extractingTweet(3000,hashtag,nN(first_day),nN(first_month),nN(final_day),nN(final_month));
//System.out.println(p.size());
//int limit = 0;	
//for(int i = 0; i < p.size(); i++)
//{
//String s = p.elementAt(i);
////System.out.println(s);
//if(!s.contains("http") && !s.contains("RT") && isValid(s) && !s.contains("pic") && !s.contains(".com") && !s.contains("news"))
//{
//	limit++;
//	writeToFile(limit,s,hashtag);
//}
//if(limit>250)
//{
//	System.out.println("Success! 250");
//	break;
//}
//}




