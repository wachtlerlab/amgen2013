function drawMunsellPatch(munsell, map)

rgb = hue_map(munsell, map);

patch = ones(100, 100, 3);

for c = 1:3
    patch(:,:,c) = patch(:,:,c) * rgb(c);
end

patch = patch/255.0;

figure('Name', munsell);
imagesc(patch);
axis image off;

end

