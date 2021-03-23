///////////////////////////////////////////////////////////
//
//
//  photochem.cpp
//
// 
// 
//
//
//
///////////////////////////////////////////////////////////

#define EIGEN_RUNTIME_NO_MALLOC

#include <cassert>
#include "aiolos.h"


/*

This requires 8-12 species that we track, besides neutral H/He.
The couplings between high-energy radiation, ionization and heating happens in this routine.

For this we assume a fixed ordering and numbering of species, as follows:
0   1   2   3   4    5    6       7    8     9    10    11
H0  H+  e-  He  He+  H++  He 23S  H2   H2+   H3+  HeH+  H-

Heating and cooling rates according to Black (1981), the physical state of primordial intergalactic clouds*/

void c_Sim::do_photochemistry() {
    
    //cout<<" Doing photochemistry "<<endl;
    
    for(int b=0; b<num_bands; b++) {
        
        if(l_i[b+1] < 0.09116) { //High-energy band, we have to redo the dS calculation and interate with electrons
            
            for(int j = num_cells + 1; j>0; j--) {
                
                double absorbed_photons;
                double convergence_epsilon = 999.;
                double last_neutrals;
                double gamma = 0;
                double neutrals    = species[0].prim[j].number_density;
                double ions        = species[1].prim[j].number_density;
                double electrons   = species[2].prim[j].number_density;
                double total       = neutrals + ions;
                
                double ionfraction = ions/neutrals;
                double x_eq;
                double number_dens_array[3] = {neutrals, ions, electrons};
                double mass_array[3] = {species[0].mass_amu*amu, species[1].mass_amu*amu, species[2].mass_amu*amu};
                
                
//                double recombination= 2.753e-14 * pow(315614./species[0].prim[j].temperature, 1.5) * pow(1. + pow(115188./species[0].prim[j].temperature, 0.407), -2.242 );
                
                double recombination = 2.59e-13 * pow(species[0].prim[j].temperature/1e4, -0.7);  //Mellema+2006 between Eqs. 11 and 12
                
                double lgT  = std::log(species[0].prim[j].temperature * K_to_eV);
                double arr_coll[9] = {-3.271396786e1, 1.35365560e1, -5.73932875e0,  1.56315498e0,  -2.87705600e-1, 3.48255977e-2, -2.63197617e-3, 1.11954395e-4, -2.03914985e-6};
                double collisions = 0;
                
                for(int k = 0; k < 9; k++)
                    collisions += arr_coll[k] * pow(lgT, k);
                collisions = std::exp(collisions);
                
                double photonenergy;
                double dtau;
                double timescale, timestep;
                
                int maxiter = 1e2;
                int iter = 0;
                //S_band(j,b)  = solar_heating(b) * std::exp(-const_opacity_solar_factor*radial_optical_depth_twotemp(j,b));
                
                /*
                for(int s=0; s<num_species; s++)
                            species[s].dS(j)  = 0.;
                */
                while(convergence_epsilon > 1e-6 && iter < maxiter) {
                    
                    last_neutrals = number_dens_array[0];
                    
                    total_opacity_twotemp(j,b)        = 0 ;
                    radial_optical_depth_twotemp(j,b) = 0.;
                    /*
                    for(int s=0; s<num_species; s++) {
                        total_opacity_twotemp(j,b)  += species[s].opacity_twotemp(j,b) * species[s].u[j].u1 ;
                    }*/
                    
                    //dtau = (1.-ionfraction) * neutrals * 2.3 * species[0].opacity_twotemp(j,b) * dx[j] ;
                    
                    dtau = 0;
                    for(int s=0; s<num_species; s++) 
                        dtau += species[s].opacity_twotemp(j,b) * number_dens_array[s] * mass_array[s] * dx[j];
                    
                    //dtau = cell_optical_depth_twotemp(j,b);
                    
                    photonenergy = h_planck*c_light/l_i12[b];
                    gamma        = S_band(j,b) / photonenergy * (l_i[b+1]-l_i[b]) * (1.-std::exp(-dtau) ) / vol[j] / 4. / (number_dens_array[0] + 0.e-4*total) * 1e18;  
                    //collisions   = 1e-5; //* 1.27e-21 * pow(species[0].prim[j].temperature, 0.5) * std::exp(-157809./species[0].prim[j].temperature) ;
                    
                    //recombination= 2.59e13  * pow(species[0].prim[j].temperature/1.e4, -0.7) * number_dens_array[1] /(total);
                    
                    x_eq = (gamma + collisions * number_dens_array[2]) / (gamma + number_dens_array[2] * (collisions + recombination));
                    //ionfraction  = (gamma + collisions * number_dens_array[2]) / (gamma + number_dens_array[2] * (collisions + recombination));
                    timescale    = 1./ (gamma + number_dens_array[2] * (collisions + recombination));
                    timestep = 1e-10; //10.*dt
                    
                    ionfraction  = x_eq + (ionfraction - x_eq) * (1.-std::exp(-timestep/timescale) )*timescale/timestep; 
                    
                    number_dens_array[0]  = (1. - ionfraction) * total;
                    number_dens_array[1]  = ionfraction * total;
                    number_dens_array[2]  = ionfraction * total;
                    convergence_epsilon = std::abs(number_dens_array[0] - last_neutrals) / total;
                    
                    if(debug >= 2) {
                        cout<<" in cell ["<<j<<"] band ["<<b<<"] gamma = "<<gamma<<" reco = "<<recombination<<" coll "<<collisions<<" dtau = "<<dtau<<" x = "<<ionfraction<<" el "<<number_dens_array[2]<<" rho/n*m = "<<species[0].u[j].u1<<"/"<<number_dens_array[0]*2.3<<" S = "<<S_band(j,b)<<endl;
                    }
                        
                    iter++;
                    
                }
                //cout<<" band "<<b<<" cell j = "<<j<<" x_final = "<<ionfraction<<" remaining high-energy flux = "<<S_band(j,b)<<endl;
                //char stopchar;
                //cin>>stopchar;
                if(debug >= 2) {
                    cout<<" cell ["<<j<<"] band ["<<b<<"] gamma = "<<gamma<<" reco = "<<recombination<<" coll "<<collisions<<" dtau = "<<dtau<<" x = "<<ionfraction<<" el "<<number_dens_array[2]<<" rho, n*m, iter = "<<species[0].u[j].u1<<"/"<<(number_dens_array[0]*species[0].mass_amu)<<"/"<<iter<<" S = "<<S_band(j,b)<<endl;                    
                }
                    
                double cooling_rec  = 1.38e-16  * species[0].prim[j].temperature * recombination * number_dens_array[1] * number_dens_array[2];
                double cooling_coll = 2.179e-11 * collisions * number_dens_array[0] * number_dens_array[2];
                
                
                if(j<num_cells+1)
                    dS_band(j,b) = gamma * number_dens_array[0] - cooling_coll - cooling_rec; //No explicit free-free & ly-alpha for now
                else
                    dS_band(j,b) = 0;
                
                //species[0].dS(j)  += dS_band(j,b) * species[s].fraction_total_opacity(j,b);
                    
                //Temporary heating fraction
                species[0].dS(j)  += 0.; //dS_band(j,b) * (1-ionfraction);
                species[1].dS(j)  += dS_band(j,b) * ionfraction * 0.01;
                species[2].dS(j)  += dS_band(j,b) * ionfraction * 0.99;
                
                if(j==num_cells+1) 
                    radial_optical_depth_twotemp(j,b) = 0.;
                
                else
                    radial_optical_depth_twotemp(j,b) = radial_optical_depth_twotemp(j+1,b) + dtau;
                
                //Update densities and hydro variables
                for(int s=0; s<num_species; s++) {
                    
                    //AOS_prim pr(u1r, u2r, u3r) ;
                    //eos->compute_conserved(&pr, &SHOCK_TUBE_UR, 1) ;
                    //or
                    double rho  = number_dens_array[s] * species[s].mass_amu * amu;
                    double newp = species[s].u[j].u2 * rho / species[s].u[j].u1;
                    double newe = species[s].u[j].u3 * rho / species[s].u[j].u1;
                    species[s].u[j] = AOS(rho, newp, newe) ;
                    
                }
                
                //species[s].prim[j].number_density = number_dens_array[s];
            }

            for(int s=0; s<num_species; s++) {
                species[s].eos->compute_primitive(&(species[s].u[0]), &(species[s].prim[0]), num_cells+1) ;
                species[s].eos->compute_auxillary(&(species[s].prim[0]), num_cells+1) ;
            }
            
            if(debug > 1) {
                cout<<"Final densities for species: "<<endl;
                
                for(int s=0; s<num_species; s++) {
                    cout<<" s= "<<s<<" n[2]= "<<species[s].prim[2].number_density<<" n[-2]= "<<species[s].prim[num_cells].number_density<<" mass_amu= "<<species[s].mass_amu<<endl;
                }
                
                char stopchar;
                cin>>stopchar;
            }
        }
    }
    
    //cout<<endl;

}
