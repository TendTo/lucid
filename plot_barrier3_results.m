% Plot BARRIER 3 results
% Copyright: Oliver Schön, 2025

%% Barrier Plots: Simple vs Complex Barrier (Num. Frequencies)
% (a) Simple robustified barrier
% num_frequencies: 6 (M=35), oversample_factor: 40, epsilon: 0.001
file = load("jair-barrier3-simple-robust2" + ".mat", "df");
fig = plotBarrierBarr3(file.df);
savefig(fig, "CSBarr3_barrier_simple_lucid.fig")

% (b) Complex robustified barrier 
% num_frequencies: 10 (M=99), oversample_factor: 40, epsilon: 0.001
file = load("jair-barrier3-complex-robust" + ".mat", "df");
fig = plotBarrierBarr3(file.df);
savefig(fig, "CSBarr3_barrier_complex_lucid.fig")

%% Ablation Study: Safety Prob. Over Num. Frequencies and Oversampling
filename = "jair-barrier3-results";

N = 112; % Number of experiments
dim = 2; % System dimensionality

satProb = [];
numFreqs = [];
Nhat = [];
oversampleFctr = [];
duration = []; % in seconds
for i = 1:N
    file = load(filename + "-" + num2str(i-1) + ".mat", "df");
    if (file.df.epsilon ~= 0 || file.df.num_frequencies <= 5) % Exluded runs
        continue
    end
    satProb = [satProb, file.df.percentage];
    numFreqs = double([numFreqs, file.df.num_frequencies]);
    Nhat = double([Nhat, file.df.lattice_resolution]);
    oversampleFctr = [oversampleFctr, file.df.oversample_factor];
    seconds = str2num(file.df.time(end-1:end));
    minutes = str2num(file.df.time(1:end-3));
    duration = [duration, seconds + 60*minutes];
end
% numFreqs = double(numFreqs);
% Nhat = (2.*numFreqs+1).*oversampleFctr;
M = numFreqs.^2-1;

%% Plot scatter
% figure
% scatter3(numFreqs, oversampleFctr, satProb, 'filled', "CData", satProb)
% ylabel("$\hat N$", "Interpreter", "latex")
% xlabel("$M$", "Interpreter", "latex")
% zlabel("$p_N$", "Interpreter", "latex")
% colormap(flipud(colormap("spring")))

% Interpolate data
% M_ = (6^2-1):(11^2-1);
numFreqs_ = 6:11;
% numFreqs_ = sqrt(M_ + 1);
Nhat_ = 0:50:800;%(2*M_+1).*(10:5:45);
% oversampleFctr_ = linspace(10, 45, 10);
% satProb_ = 0:5:100;
[numFreqs_, Nhat_] = meshgrid(numFreqs_, Nhat_);
satProb_inter = griddata(numFreqs, Nhat, satProb, numFreqs_, ...
    Nhat_, "v4");
satProb_inter(satProb_inter<0) = 0;
duration_inter = griddata(numFreqs, Nhat, duration, numFreqs_, ...
    Nhat_, "v4");
duration_inter(duration_inter<0) = 0;

% Plot safety probability
figure
contourf(numFreqs_, Nhat_, satProb_inter, [0,5,15,25,30,35,40,45,50,55], ...
    '--', "ShowText", true, "LabelFormat", "%0.1f %%", "FaceAlpha", 0.7)
colormap(flipud(colormap("spring")))
ax = gca;
ax.XAxis.TickValues = 6:11;
ax.XAxis.TickLabels = ax.XAxis.TickValues.^2-1;
% ax.YAxis.TickValues = 10:5:45;
% ax.YAxis.TickLabels = ax.YAxis.TickLabels;
% ylabel("oversamplefctr", "Interpreter", "latex")
% xlabel("number of frequencies", "Interpreter", "latex")
ylabel("$\sqrt{\hat{N}}$", "Interpreter", "latex")
xlabel("$M$", "Interpreter", "latex")
zlabel("$p_N$", "Interpreter", "latex")
% papers_layout(gcf, 6, 6) % Nice, legible single column in JAIR

% Plot duration
figure
contourf(numFreqs_, Nhat_, duration_inter/60, [0, 1, 2, 5, 15, 30, 60, 120], ...satProb_, ...
    '--', "ShowText", true, "LabelFormat", "%0.1f min", "FaceAlpha", 0.7)
colormap(flipud(colormap("spring")))
ax = gca;
ax.XAxis.TickValues = 6:11;
ax.XAxis.TickLabels = ax.XAxis.TickValues.^2-1;
% ax.YAxis.TickValues = 10:5:45;
% ax.YAxis.TickLabels = ax.YAxis.TickLabels;
% ylabel("oversamplefctr", "Interpreter", "latex")
% xlabel("number of frequencies", "Interpreter", "latex")
ylabel("$\sqrt{\hat{N}}$", "Interpreter", "latex")
xlabel("$M$", "Interpreter", "latex")
zlabel("$p_N$", "Interpreter", "latex")
% papers_layout(gcf, 6, 6) % Nice, legible single column in JAIR


%% Function definitions used

function fig = plotBarrierBarr3(df)
% Plots the barrier for the experiment file df
X_limits = [df.X_bounds_lower', df.X_bounds_upper'];
regions_init = { ...
    [1, 2; -0.7, 0.3], ...
    [-1.8, -1.4; -0.1, 0.1], ...
    [-1.4, -1.2; -0.5, 0.1]};
regions_unsafe = { ...
    [0.4, 0.6; 0.2, 0.6], ...
    [0.6, 0.7; 0.2, 0.4]};
eta = df.eta;
gamma = df.gamma;


res = sqrt(length(df.x_lattice)); % lattice size per dim
x1 = reshape(df.x_lattice(:, 1), res, res);
x2 = reshape(df.x_lattice(:, 2), res, res);
b = reshape(df.x_barrier_values, res, res);

fprintf("Plotting barrier with sat. prob. %0.5f%%\n", df.percentage)
fig = figure;
h_barr = surf(x1, x2, b, 'FaceAlpha', .70, 'EdgeAlpha', .0, ...
    "DisplayName", "$B(x)$");
colormap(flipud(colormap("spring")))
view(20, 60)
grid on
xlabel("$x_1$", 'Interpreter', 'latex')
ylabel("$x_2$", 'Interpreter', 'latex')
zlabel("$B(x)$", 'Interpreter', 'latex')
hold on

% Regions
ax = gca;
tmp = combvec(X_limits(1, :), X_limits(2, :));
patch(tmp(1, [1, 2, 4, 3]), tmp(2, [1, 2, 4, 3]), 'k', 'FaceAlpha', .0)
p_init = cell(1, length(regions_init));
for i = 1:length(regions_init)
    tmp = combvec(regions_init{i}(1, :), regions_init{i}(2, :));
    patch(tmp(1, [1, 2, 4, 3]), tmp(2, [1, 2, 4, 3]), 'b', 'FaceAlpha', 1)
    p_init{i} = ax.Children(1);
end
p_unsafe = cell(1, length(regions_unsafe));
for i = 1:length(regions_unsafe)
    tmp = combvec(regions_unsafe{i}(1, :), regions_unsafe{i}(2, :));
    patch(tmp(1, [1, 2, 4, 3]), tmp(2, [1, 2, 4, 3]), 'r', 'FaceAlpha', 1)
    p_unsafe{i} = ax.Children(1);
end

% Level sets
patch([X_limits(1, 1) X_limits(1, 2) X_limits(1, 2) X_limits(1, 1)], ...
    [X_limits(2, 1) X_limits(2, 1) X_limits(2, 2) X_limits(2, 2)], ...
    eta * ones(1, 4), 'blue', 'FaceAlpha', .2, 'EdgeColor', 'blue')
p_eta = ax.Children(1);
patch([X_limits(1, 1) X_limits(1, 2) X_limits(1, 2) X_limits(1, 1)], ...
    [X_limits(2, 1) X_limits(2, 1) X_limits(2, 2) X_limits(2, 2)], ...
    gamma * ones(1, 4), 'red', 'FaceAlpha', .2, 'EdgeColor', 'red')
p_gamma = ax.Children(1);
% [~, h] = contour(x1, x2, b);
% h.LevelList = [eta, gamma];
% h.LineStyle = '--';
% h.EdgeColor = "k";
% h.FaceAlpha = .3;
% h.LineWidth = 2;
[~, h] = contour(x1, x2, b);
h.LevelList = [eta];
h.LineStyle = '--';
h.EdgeColor = "k";
h.LineWidth = 2;
h.ZLocation = eta;
[~, h] = contour(x1, x2, b);
h.LevelList = [gamma];
h.LineStyle = '--';
h.EdgeColor = "k";
h.LineWidth = 2;
h.ZLocation = gamma;
[~, h] = contour(x1, x2, b);
h.LevelList = 0;
h.EdgeColor = "r";
h.LineWidth = 2;
hold off
lt = light("Style", "Infinite");

papers_layout(gcf, 6, 6) % Nice, legible single column in JAIR
end