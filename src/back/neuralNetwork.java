/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    MessageClassifier.java
 *    Copyright (C) 2009 University of Waikato, Hamilton, New Zealand
 *
 */

package back;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.Serializable;
import java.util.ArrayList;

/**
 * Java program for classifying short text messages into two classes 'miss'
 * and 'hit'.
 * <p/>
 * See also wiki article <a href="http://weka.wiki.sourceforge.net/MessageClassifier">MessageClassifier</a>.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class neuralNetwork
  implements Serializable {

  /** for serialization. */
  private static final long serialVersionUID = -123455813150452885L;

  /** The training data gathered so far. */
  private Instances m_Data = null;

  /** The filter used to generate the word counts. */
  private StringToWordVector m_Filter = new StringToWordVector();

  /** The actual classifier. */
  private Classifier m_Classifier = null;

  /** Whether the model is up to date. */
  private boolean m_UpToDate;

  /**
   * Constructs empty training dataset.
   */
  public neuralNetwork() {
    
    m_Classifier = new MultilayerPerceptron();

    // Create vector of attributes.
    ArrayList<Attribute> attributes = new ArrayList<Attribute>(3+3);
    //NB NBM RF SVM LD classe

    // Add attribute for holding messages.
    attributes.add(new Attribute("NB", (String) null));
    attributes.add(new Attribute("NBM",(String) null));
    attributes.add(new Attribute("RF",(String) null));
    attributes.add(new Attribute("SVM",(String) null));
    attributes.add(new Attribute("LD",(String) null));

    // Add class attribute.
    ArrayList<String> classValues = new ArrayList<String>(2);
    classValues.add("Hillary");
    classValues.add("Trump");
    attributes.add(new Attribute("Class", classValues));

    // Create dataset with initial capacity of 100, and set index of class.
    m_Data = new Instances("MLP", attributes, 200);
    m_Data.setClassIndex(2+3);
  }

  /**
   * Updates model using the given training message.
   *
   * @param message	the message content
   * @param classValue	the class label
   */
  public void updateData(String nb,String nbm, String rf, String svm, String ld, String classValue) {
    // Make message into instance.
    Instance instance = makeInstance(nb,nbm,rf,svm,ld, m_Data);

    // Set class value for instance.
    instance.setClassValue(classValue);

    // Add instance to training data.
    m_Data.add(instance);

    m_UpToDate = false;
  }

  /**
   * Classifies a given message.
   *
   * @param message	the message content
 * @return 
   * @throws Exception 	if classification fails
   */
  public String classifyMessage(String nb,String nbm, String rf,String svm,String ld) throws Exception {
    // Check whether classifier has been built.
    if (m_Data.numInstances() == 0)
      throw new Exception("No classifier available.");

    // Check whether classifier and filter are up to date.
    if (!m_UpToDate) {
      // Initialize filter and tell it about the input format.
      m_Filter.setInputFormat(m_Data);

      // Generate word counts from the training data.
      Instances filteredData  = Filter.useFilter(m_Data, m_Filter);

      // Rebuild classifier.
      m_Classifier.buildClassifier(filteredData);

      m_UpToDate = true;
    }

    // Make separate little test set so that message
    // does not get added to string attribute in m_Data.
    Instances testset = m_Data.stringFreeStructure();

    // Make message into test instance.
    Instance instance = makeInstance(nb, nbm, rf, svm, ld, testset);

    // Filter instance.
    m_Filter.input(instance);
    Instance filteredInstance = m_Filter.output();

    // Get index of predicted class value.
    double predicted = m_Classifier.classifyInstance(filteredInstance);

    // Output class value.
    //System.err.println("Message classified as : " +
	//	       m_Data.classAttribute().value((int) predicted));
    
    return m_Data.classAttribute().value((int) predicted);
  }

  private Instance makeInstance(String nb,String nbm, String rf,String svm,String ld, Instances data) {
    // Create instance of length two.
    Instance instance = new DenseInstance(2*3);

    // Set value for message attribute
    Attribute messageAtt = data.attribute("NB");
    instance.setValue(messageAtt, messageAtt.addStringValue(nb));
    messageAtt = data.attribute("NBM");
    instance.setValue(messageAtt, messageAtt.addStringValue(nbm));
    messageAtt = data.attribute("RF");
    instance.setValue(messageAtt, messageAtt.addStringValue(rf));
    messageAtt = data.attribute("SVM");
    instance.setValue(messageAtt, messageAtt.addStringValue(svm));
    messageAtt = data.attribute("LD");
    instance.setValue(messageAtt, messageAtt.addStringValue(ld));

    // Give instance access to attribute information from the dataset.
    instance.setDataset(data);

    return instance;
  }
}