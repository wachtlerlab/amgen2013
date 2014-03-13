r_x = linss2_10e_1(1:4:389,1);
r_l = linss2_10e_1(1:4:389,2);
r_m = linss2_10e_1(1:4:389,3);
r_s = linss2_10e_1(1:4:389,4);
        
        
plot(r_x,r_l); hold on;
plot(r_x,r_m);
plot(r_x,r_s);
set(gca, 'YTickLabelMode', 'manual', 'YTickLabel', []);
text([590], [0.95], ['\rm L'], 'Fontsize',14)
text([510], [0.95], ['\rm M'], 'Fontsize',14)
text([451], [0.95], ['\rm S'], 'Fontsize',14)
xlabel('\lambda [nm]');