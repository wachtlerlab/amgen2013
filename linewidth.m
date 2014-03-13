magic = e_magic .* s_magic .* r_m;
h=plot(r_x,magic, 'k');
set(h,'LineWidth',3);
axis([380, 790, 0, 0.8]);
set(gca,'LooseInset',get(gca,'TightInset'));
set(gca, 'YTickLabel',[]);
xlabel('\lambda [nm]','Fontsize',18);
print('-deps','-r600','v_111.eps');