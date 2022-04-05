//  MOEAD_main.java
//
//  Author:
//       Antonio J. Nebro <antonio@lcc.uma.es>
//       Juan J. Durillo <durillo@lcc.uma.es>
//
//  Copyright (c) 2011 Antonio J. Nebro, Juan J. Durillo
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
// 
//  You should have received a copy of the GNU Lesser General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

package jmetal.metaheuristics.moead;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.SolutionSet;
import jmetal.operators.crossover.CrossoverFactory;
import jmetal.operators.mutation.MutationFactory;
import jmetal.problems.Kursawe;
import jmetal.problems.ProblemFactory;
import jmetal.qualityIndicator.QualityIndicator;
import jmetal.util.Configuration;
import jmetal.util.JMException;

import java.io.IOException;
import java.util.HashMap;
import java.util.logging.FileHandler;
import java.util.logging.Logger;

/**
 * This class executes the algorithm described in:
 * H. Li and Q. Zhang,
 * "Multiobjective Optimization Problems with Complicated Pareto Sets,  MOEA/D
 * and NSGA-II". IEEE Trans on Evolutionary Computation, vol. 12,  no 2,
 * pp 284-302, April/2009.
 */
public class MOEAAD_term1 {
    public static Logger logger_;      // Logger object
    public static FileHandler fileHandler_; // FileHandler object

    /**
     * @param args Command line arguments. The first (optional) argument specifies
     *             the problem to solve.
     * @throws JMException
     * @throws IOException
     * @throws SecurityException      Usage: three options
     *                                - jmetal.metaheuristics.moead.MOEAD_main
     *                                - jmetal.metaheuristics.moead.MOEAD_main problemName
     *                                - jmetal.metaheuristics.moead.MOEAD_main problemName ParetoFrontFile
     * @throws ClassNotFoundException
     */
    public static void main(String[] args) throws JMException, SecurityException, IOException, ClassNotFoundException {
        Problem problem;         // The problem to solve
        Algorithm algorithm;         // The algorithm to use
        Operator crossover;         // Crossover operator
        Operator mutation;         // Mutation operator

        QualityIndicator indicators; // Object to get quality indicators

        HashMap parameters; // Operator parameters

        // Logger object and file to store log messages
        logger_ = Configuration.logger_;
        fileHandler_ = new FileHandler("MOEAD.log");
        logger_.addHandler(fileHandler_);

        indicators = null;
        String[] allps = {"DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4",
                            "WFG1", "WFG2", "WFG4", "WFG5", "WFG6", "WFG7", "WFG8", "WFG9"}; 
        int[] nol = {3,5,10,15};
        
//        for(int itp=0; itp<4; itp++) {
         for(int itp=4; itp<allps.length; itp++) {
                
            for(int nitr=0; nitr<nol.length; nitr++) {
                for(int sr=0; sr<31; sr++) {
                    int noo = nol[nitr];
                    int nvr = noo-1 + 20;
                    int nkr = 2*(noo-1);
                    int nlr = 20; 

//                    Object[] params = { "Real", nvr, noo};
                     Object[] params = { "Real", nkr, nlr, noo};
                    String pname = allps[itp];
                    problem = (new ProblemFactory()).getProblem(pname, params);
                    algorithm = new MOEAAD(problem);
                    algorithm.setInputParameter("normalize", true);
                    int pops = 0;
                    if(noo==3) {pops=105;}
                    else if(noo==5){pops=210;}
                    else if(noo==10){pops=275;}
                    else if(noo==15){pops=135;}
                    algorithm.setInputParameter("populationSize", pops);
                    int ngen = 500;
                    if(itp==0){if(noo==3)ngen=661;
                                else if(noo==5)ngen=765;
                                else if(noo==10)ngen=1003;
                                else if(noo==15)ngen=1123; }
                    else if(itp==1){if(noo==3)ngen=733;
                                else if(noo==5)ngen=677;
                                else if(noo==10)ngen=854;
                                else if(noo==15)ngen=1102; }
                    else if(itp==2){if(noo==3)ngen=624;
                                else if(noo==5)ngen=708;
                                else if(noo==10)ngen=921;
                                else if(noo==15)ngen=1027; }
                    else if(itp==3){if(noo==3)ngen=1102;
                                else if(noo==5)ngen=914;
                                else if(noo==10)ngen=900;
                                else if(noo==15)ngen=732; }
                    else if(itp==4){if(noo==3)ngen=1014;
                                else if(noo==5)ngen=1161;
                                else if(noo==10)ngen=2575;
                                else if(noo==15)ngen=1597; }
                    else if(itp==5){if(noo==3)ngen=1004;
                                else if(noo==5)ngen=1082;
                                else if(noo==10)ngen=1537;
                                else if(noo==15)ngen=1374; }
                    else if(itp==6){if(noo==3)ngen=793;
                                else if(noo==5)ngen=731;
                                else if(noo==10)ngen=797;
                                else if(noo==15)ngen=937; }
                    else if(itp==7){if(noo==3)ngen=737;
                                else if(noo==5)ngen=682;
                                else if(noo==10)ngen=737;
                                else if(noo==15)ngen=784; }
                    else if(itp==8){if(noo==3)ngen=755;
                                else if(noo==5)ngen=675;
                                else if(noo==10)ngen=725;
                                else if(noo==15)ngen=746; }
                    else if(itp==9){if(noo==3)ngen=717;
                                else if(noo==5)ngen=706;
                                else if(noo==10)ngen=760;
                                else if(noo==15)ngen=841; }
                    else if(itp==10){if(noo==3)ngen=692;
                                else if(noo==5)ngen=675;
                                else if(noo==10)ngen=709;
                                else if(noo==15)ngen=771; }
                    else if(itp==11){if(noo==3)ngen=671;
                                else if(noo==5)ngen=723;
                                else if(noo==10)ngen=757;
                                else if(noo==15)ngen=778; }
                    algorithm.setInputParameter("dataDirectory",
                        "E:\\Work\\MSU\\papers\\LHFiD\\first_submission\\codes\\MOEA-AD\\EMOStudy_jMetal-master\\EMOStudy_jMetal-master\\weight");
                    algorithm.setInputParameter("maxEvaluations", (ngen+1)*pops);
                    algorithm.setInputParameter("T", 20);
                    algorithm.setInputParameter("delta", 0.9);
                    algorithm.setInputParameter("nr1", 1);
                    algorithm.setInputParameter("nr2", 2);
                    algorithm.setInputParameter("functionType1", "_PBI");
                    algorithm.setInputParameter("functionType2", "_TCHA");
                    
                    // Crossover operator
                    parameters = new HashMap();
                    parameters.put("probability", 0.9);
                    parameters.put("distributionIndex", 20.0);
                    crossover = CrossoverFactory.getCrossoverOperator("SBXCrossover",
                                    parameters);

                    // Mutation operator
                    parameters = new HashMap();
                    parameters.put("probability", 1.0 / problem.getNumberOfVariables());
                    parameters.put("distributionIndex", 20.0);
                    mutation = MutationFactory.getMutationOperator("PolynomialMutation", parameters);

                    algorithm.addOperator("crossover", crossover);
                    algorithm.addOperator("mutation", mutation);
                    
                    SolutionSet population = algorithm.execute();
                    System.out.println("FUN_"+pname+"_at_"+noo+"_"+sr+"\t"+ngen);
                    population.printObjectivesToFile("Term1/"+"FUN_"+pname+"_at_"+noo+"_"+sr+"_"+"FT");

                    
        

                } // if
            } //main
        }
    }
} // MOEAD_main
