import request from 'supertest';
import { expect } from 'chai';
import app from '../app';

describe('GET /', function() {
  it('应该返回状态码 200', function(done) {
    request(app)
      .get('/')
      .expect(200)
      .end(done);
  });
});