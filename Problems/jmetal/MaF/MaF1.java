/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jmetal.problems.MaF;

import jmetal.core.Problem;
import jmetal.encodings.solutionType.BinaryRealSolutionType;
import jmetal.encodings.solutionType.RealSolutionType;

import java.util.Random;
import jmetal.core.Solution;
import jmetal.core.Variable;
import jmetal.util.JMException;

/**
 *
 * @author sarang
 */
public class MaF1 extends Problem
{
    public MaF1(String solutionType) throws ClassNotFoundException {
        this(solutionType, 7, 3);
    } // DTLZ1

    /**
     * Creates a MaF1 problem instance
     * 
     * @param numberOfVariables
     *            Number of variables
     * @param numberOfObjectives
     *            Number of objective functions
     * @param solutionType
     *            The solution type must "Real" or "BinaryReal".
     */
    public MaF1(String solutionType, Integer numberOfVariables, Integer numberOfObjectives) 
    {
        numberOfVariables_ = numberOfVariables;
        numberOfObjectives_ = numberOfObjectives;
        numberOfConstraints_ = 0;
        problemName_ = "MaF1";

        lowerLimit_ = new double[numberOfVariables_];
        upperLimit_ = new double[numberOfVariables_];
        for (int var = 0; var < numberOfVariables; var++) {
                lowerLimit_[var] = 0.0;
                upperLimit_[var] = 1.0;
        } // for

        if (solutionType.compareTo("BinaryReal") == 0)
                solutionType_ = new BinaryRealSolutionType(this);
        else if (solutionType.compareTo("Real") == 0)
                solutionType_ = new RealSolutionType(this);
        else {
                System.out.println("Error: solution type " + solutionType
                                + " invalid");
                System.exit(-1);
        }
    }

    @Override
    public void evaluate(Solution solution) throws JMException
    {
        Variable[] gen = solution.getDecisionVariables();

        double[] x = new double[numberOfVariables_];
        double[] f = new double[numberOfObjectives_];
        int k = numberOfVariables_ - numberOfObjectives_ + 1;

        for (int i = 0; i < numberOfVariables_; i++)
                x[i] = gen[i].getValue();

        double g = 0.0;
        for (int i = numberOfObjectives_ - 1; i < numberOfVariables_; i++)
            g += (x[i] - 0.5) * (x[i] - 0.5);

        for (int i = 0; i < numberOfObjectives_; i++)
            f[i] = (1.0 + g);

        for (int i = 0; i < numberOfObjectives_; i++) {
            int m = i+1;
            if (m==1) {
                double prod = 1.0;
                for (int j = 0; j < numberOfObjectives_ - m - (i + 1); j++)
                    prod *= x[j];
                f[i] *= (1.0 - prod);
            }
            else if(m < numberOfObjectives_){
                double prod = 1.0;
                for (int j = 0; j < numberOfObjectives_ - m - (i + 1); j++)
                    prod *= x[j];
                prod *= (1.0 - x[numberOfObjectives_ - m]); 
                f[i] *= (1.0 - prod);
            }
            else{
                f[i] *= x[0];
            }
        }// for

        for (int i = 0; i < numberOfObjectives_; i++)
            solution.setObjective(i, f[i]);
    }
    
}
