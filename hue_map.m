function [ rgb ] = hue_map(munsell, map)

hue_spec = munsell(1);
hue = munsell(2:3);
value = munsell(4:5);
chroma = munsell(6:7);

if strcmp(hue_spec, 'A')
    hue_spec = 2.5;
elseif strcmp(hue_spec, 'B')
    hue_spec = 5.0;
elseif strcmp(hue_spec, 'C')
    hue_spec = 7.5; 
elseif strcmp(hue_spec, 'D')
    hue_spec = 10.0;
elseif strcmp(hue_spec, 'E')
    hue_spec = 1.25;
elseif strcmp(hue_spec, 'F')
    hue_spec = 3.75;
elseif strcmp(hue_spec, 'G')
    hue_spec = 6.25;
elseif strcmp(hue_spec, 'H')
    hue_spec = 8.75;
elseif strmcp(hue_spec, 'NEUT')
    hue_spec = 0;
end

if hue(1) == hue(2)
    hue = hue(1);
end

hue_str = sprintf('%.2f%s', hue_spec, hue);
value = str2double(value)/10;
chroma = str2double(chroma);


%find the index of the row with value, chroma, and hue_str
index = find (map.Value == value & map.Chroma == chroma & cellfun(@(x) strcmp(x, hue_str), map.Hue));

%now get R, G, B values from the row at index
rgb = [map.R(index), map.G(index), map.B(index)];

end