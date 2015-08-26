package edu.utaustin.ensemble;

import libsvm.svm_parameter;
import weka.classifiers.*;
import weka.classifiers.functions.LibSVM;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class SVMClassifier {
	String[] prediction;
	static List<String> slotNames;


	
	public void classify(String train_file, String test_file) throws Exception{
		slotNames = Arrays.asList(
				"per:date_of_birth",
				"per:age",
				"per:country_of_birth",
				"per:stateorprovince_of_birth",
				"per:city_of_birth",
				"per:date_of_death",
				"per:country_of_death",
				"per:stateorprovince_of_death",
				"per:city_of_death",
				"per:cause_of_death",
				"per:religion",
				"org:number_of_employees_members",
				"org:date_founded",
				"org:date_dissolved",
				"org:country_of_headquarters",
				"org:stateorprovince_of_headquarters",
				"org:city_of_headquarters",
				"org:website",

				"per:alternate_names",
				"per:origin",
				"per:countries_of_residence",
				"per:statesorprovinces_of_residence",
				"per:cities_of_residence",
				"per:schools_attended",
				"per:title",
				"per:employee_or_member_of",
				"per:spouse",
				"per:children",
				"per:parents",
				"per:siblings",
				"per:other_family",
				"per:charges",
				"org:alternate_names",
				"org:political_religious_affiliation",
				"org:top_members_employees",
				"org:members",
				"org:member_of",
				"org:subsidiaries",
				"org:parents",
				"org:founded_by",
				"org:shareholders",
				"per:awards_won",
				"per:charities_supported",
				"per:diseases",
				"org:products",
				"per:pos-from",
				"per:neg-from",
				"per:pos-towards",
				"per:neg-towards",
				"org:pos-from",
				"org:neg-from",
				"org:pos-towards",
				"org:neg-towards",
				"gpe:pos-from",
				"gpe:neg-from",
				"gpe:pos-towards",
				"gpe:neg-towards",

				"org:employees_or_members",
				"gpe:employees_or_members",

				"org:students",
				"gpe:births_in_city",
				"gpe:births_in_stateorprovince",
				"gpe:births_in_country",
				"gpe:residents_of_city",
				"gpe:residents_of_stateorprovince",
				"gpe:residents_of_country",
				"gpe:deaths_in_city",
				"gpe:deaths_in_stateorprovince",
				"gpe:deaths_in_country",

				"per:holds_shares_in",
				"org:holds_shares_in",
				"gpe:holds_shares_in",

				"per:organizations_founded",
				"org:organizations_founded",
				"gpe:organizations_founded",

				"gpe:member_of",

				"per:top_member_employee_of",
				"gpe:headquarters_in_city",
				"gpe:headquarters_in_stateorprovince",
				"gpe:headquarters_in_country");

		DataSource source = new DataSource(train_file);
		Instances data = source.getDataSet();
		//num_instance = data.numInstances();
		//all instances = data.toString()
		//get instance = data.instance(n)
//		System.out.println(data.toString());
//		System.out.println(data.instance(2));
//		LibSVM[] svms = new LibSVM[41];
//		System.exit(1);
		DataSource test = new DataSource(test_file);
		Instances data_test = test.getDataSet();
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		if (data_test.classIndex() == -1)
			data_test.setClassIndex(data.numAttributes() - 1);
		boolean multi_svms = true;

		if(multi_svms){
			HashMap<String, LibSVM> hash = new HashMap<>();
			for(int slot_num=1; slot_num<slotNames.size(); slot_num++) {
				System.out.println(slot_num);
				String slot = slotNames.get(slot_num);
				System.out.println(slot);
				Instances new_data = new Instances(data);
//				System.out.println(new_data.numInstances());
				for(int instance_num=new_data.numInstances()-1; instance_num >= 0; instance_num--){
//					System.out.println(instance_num);
//					System.out.println(new_data.instance(instance_num));
					Instance inst = new_data.instance(instance_num);
//					System.out.println(inst.toString(new_data.numAttributes() - 2));
					String instSlot = inst.toString(new_data.numAttributes() - 2);
					if(!instSlot.equals(slot)){
						new_data.delete(instance_num);
					}
				}
				System.out.println(new_data.toString());
//				System.exit(1);
				System.out.println(new_data.numAttributes());
				new_data.deleteAttributeAt(new_data.numAttributes() - 2);
				System.out.println(new_data.numAttributes());
				System.out.println(data.numAttributes());
//			System.out.println(new_data.attribute("rel"));
//			System.out.println(new_data.attribute(new_data.numAttributes()-2));
//				System.exit(1);
				if(new_data.numInstances() > 0){
					LibSVM svm = new LibSVM();
					svm.buildClassifier(new_data);
					hash.put(slot,svm);
				}
//				else{
//
//				}
			}
			prediction = new String[data_test.numInstances()];
			for (int i = 0; i < data_test.numInstances(); i++) {
				Instances new_test_data = new Instances(data_test);
				Instance test_inst = new_test_data.instance(i);
				String instSlot = test_inst.toString(data_test.numAttributes() - 2);
				new_test_data.deleteAttributeAt(new_test_data.numAttributes() - 2);
				if(hash.containsKey(instSlot)) {
//					test_inst.deleteAttributeAt(data_test.numAttributes() - 2);
					double pred = hash.get(instSlot).classifyInstance(new_test_data.instance(i));
					System.out.print("ID: " + data_test.instance(i).value(0));
					System.out.print(", actual: " + data_test.classAttribute().value((int) data_test.instance(i).classValue()));
					System.out.println(", predicted: " + data_test.classAttribute().value((int) pred));
					prediction[i] = data_test.classAttribute().value((int) pred);
				}
				else{
					prediction[i] = "w";
				}
			}
//			System.exit(1);
		}
		else {
			//InputMappedClassifier = new InputMappedClassifier();
			LibSVM svm = new LibSVM();
			//svm_parameter pre= new svm_parameter();
			// pre.kernel_type= svm_parameter.POLY;
			// pre.gamma= 3;
			//pre.degree=1;
			svm.buildClassifier(data);
			prediction = new String[data_test.numInstances()];
			for (int i = 0; i < data_test.numInstances(); i++) {
				double pred = svm.classifyInstance(data_test.instance(i));
				System.out.print("ID: " + data_test.instance(i).value(0));
				System.out.print(", actual: " + data_test.classAttribute().value((int) data_test.instance(i).classValue()));
				System.out.println(", predicted: " + data_test.classAttribute().value((int) pred));
				prediction[i] = data_test.classAttribute().value((int) pred);
			}
//		System.out.println(prediction[0]);
//		System.exit(1);
		}
	}
}
