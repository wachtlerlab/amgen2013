h=plot(r_x,e_magic, 'k');
set(h,'LineWidth',3);
axis([380, 790, 0, 1]);
set(gca,'LooseInset',get(gca,'TightInset'));
set(gca, 'YTickLabel',[]);
xlabel('\lambda [nm]','Fontsize',18);
print('-deps','-r600', 'e_magic.eps');