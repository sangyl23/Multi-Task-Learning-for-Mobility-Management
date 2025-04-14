% close all, clear all, clc;

function [x_t, y_t] = spiral_motion(x0, y0, x_mid, y_mid, omega, c, t)
    
    %{
    c = 10 / 2 / pi;
    omega = -0.5;

    x0 = round(10 / 0.2);
    y0 = round(286.8 / 0.2);
    %}

    x0 = (x0 - x_mid) * 0.2; % * 0.2 is because the spacing of grid point is 0.2 m
    y0 = (y0 - y_mid) * 0.2;
    theta0 = atan2(y0, x0); % initial angle
    r0 = sqrt(x0^2 + y0^2); % initial radius

    % omega = 0.3; % angular velocity (rad/s)
    % c = 5 / (2 * pi); % radial growth rate (m/s)
    
    % Time array
    % t = 0 : 0.16 : 1.6; % adjust time range and resolution as needed
    
    % Calculate position at time t
    theta_t = theta0 + omega * t;
    r_t = r0 + c * t;
    x_t = round(r_t .* cos(theta_t) / 0.2 + x_mid); % (t_len, 1)
    y_t = round(r_t .* sin(theta_t) / 0.2 + y_mid); % (t_len, 1)

    %{
    % Plot the spiral
    figure;
    plot(x_t * 0.2, y_t * 0.2, '<-', 'MarkerSize', 8);
    xlabel('$x$ (m)', 'FontSize', 13, 'interpreter', 'latex');
    ylabel('$y$ (m)', 'FontSize', 13, 'interpreter', 'latex');
    %title('Spiral motion ($c = -\frac{10}{2 \pi}$, $b = 0.5$)', 'FontSize', 13, 'interpreter', 'latex');
    axis equal;
    grid on;
    %}
    
end
